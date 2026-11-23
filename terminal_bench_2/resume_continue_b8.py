"""Resume the interrupted Terminal-Bench continuation run from b8.

The existing run has committed b0..b7 and has partial b8 proposals/traces.
This script completes b8 from the b7 incumbent, then runs the remaining
batches with the original G=2/R=2 configuration.
"""

from __future__ import annotations

import json
import fcntl
import os
from pathlib import Path
import shutil
import signal
import time

import tthe_bridge as bridge
import tthe_proposer as proposer
from tthe_optimize import loadable, score_observations
from safe_docker import check_daemon, ensure_network, host_state, remove_idle_network


ROOT = Path(__file__).resolve().parent
RUN = ROOT / "logs" / "full89_deepseek_tthe_continue_b6_b17_g2r2_t100_20261001_132551"
AGENTS = ROOT / "agents"
MODEL = "openai/deepseek-chat"
PROPOSER_MODEL = "deepseek-chat"
PYTHON = str(ROOT / ".venv313" / "bin" / "python")
HARBOR = str(ROOT / ".venv313" / "bin" / "harbor")
DATASET = "terminal-bench@2.0"
SYSTEM_DOCKER_HOST = "unix:///var/run/docker.sock"
PRIVATE = ROOT / ".controller-private" / RUN.name


def clear_proxy() -> None:
    for key in bridge.PROXY_ENV_VARS:
        os.environ.pop(key, None)
    os.environ["NO_PROXY"] = "*"
    os.environ["no_proxy"] = "*"


def preflight_docker() -> None:
    """Pin the existing system daemon; never start, stop or reconfigure one."""
    os.environ.pop("DOCKER_CONTEXT", None)
    os.environ["DOCKER_HOST"] = SYSTEM_DOCKER_HOST
    os.environ["TTHE_SAFE_DOCKER"] = "1"
    check_daemon()
    ensure_network()
    print(f"DOCKER_HOST={SYSTEM_DOCKER_HOST}; safe network; concurrency=1", flush=True)


def pause_after(batch: int) -> bool:
    target = os.environ.get("TTHE_PAUSE_AFTER_BATCH")
    if target and batch >= int(target):
        print(f"PAUSED after batch {batch} (TTHE_PAUSE_AFTER_BATCH={target})", flush=True)
        return True
    return False


def reusable(candidate: str, task: str, trace_dir: Path) -> bool:
    stem = f"{candidate}__{task}__a0"
    try:
        evidence = json.loads((trace_dir / f"{stem}.evidence.json").read_text())
        trajectory = json.loads((trace_dir / f"{stem}.raw_trajectory.json").read_text())
    except (OSError, json.JSONDecodeError):
        return False
    # Task/verifier failures remain evidence for the model judge, not hard gates.
    return (evidence.get("task_name") == task and bool(trajectory.get("steps"))
            and (evidence.get("status") or {}).get("exception_type")
            not in ("CancelledError", "MissingTrial"))


def private_checkpoint(candidate: str, task: str, trace_dir: Path) -> Path:
    directory = PRIVATE / trace_dir.name
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    return directory / f"{candidate}__{task}.json"


def save_observation(candidate: str, observation: bridge.TrialObservation, trace_dir: Path) -> None:
    path = private_checkpoint(candidate, observation.task_name, trace_dir)
    value = {
        "task_name": observation.task_name, "attempt": observation.attempt,
        "reward": observation.reward, "status": observation.status,
        "raw_trace_path": str(observation.raw_trace_path),
        "trace_view_path": str(observation.trace_view_path),
        "evidence_path": str(observation.evidence_path),
    }
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
    path.chmod(0o600)


def cached_observation(candidate: str, task: str, trace_dir: Path) -> bridge.TrialObservation | None:
    path = private_checkpoint(candidate, task, trace_dir)
    if not path.exists():
        return None
    value = json.loads(path.read_text())
    for key in ("raw_trace_path", "trace_view_path", "evidence_path"):
        value[key] = Path(value[key])
    return bridge.TrialObservation(**value)


def observe(candidate: str, tasks: list[str], trace_dir: Path) -> list[bridge.TrialObservation]:
    trace_dir.mkdir(parents=True, exist_ok=True)
    observations = []
    for task in tasks:
        if reusable(candidate, task, trace_dir):
            cached = cached_observation(candidate, task, trace_dir)
            if cached is not None:
                observations.append(cached)
            print(f"   OBSERVE cached {candidate} {task}", flush=True)
            continue
        attempt_dir = trace_dir / "attempts" / candidate / f"{task}-{time.time_ns()}"
        attempt_dir.mkdir(parents=True)
        # Preserve an interrupted attempt before publishing its replacement.
        interrupted = trace_dir / "interrupted" / candidate / attempt_dir.name
        for previous in trace_dir.glob(f"{candidate}__{task}__a0.*"):
            interrupted.mkdir(parents=True, exist_ok=True)
            previous.rename(interrupted / previous.name)
        checkpoint = private_checkpoint(candidate, task, trace_dir)
        if checkpoint.exists():
            history = checkpoint.parent / "history"
            history.mkdir(exist_ok=True, mode=0o700)
            checkpoint.rename(history / f"{checkpoint.stem}-{time.time_ns()}.json")
        print(f"   OBSERVE {candidate} {task}; attempt={attempt_dir}", flush=True)
        before = host_state()
        batch_obs = bridge.observe_candidate(
            candidate, f"agents.{candidate}:AgentHarness", [task], attempt_dir,
            dataset=DATASET, model=MODEL, environment="docker", concurrency=1,
            max_turns=100, harbor_command=HARBOR, timeout=14400, workdir=ROOT,
        )
        if host_state() != before:
            raise RuntimeError("Host default route/DNS/daemon identity changed; pausing")
        for observation in batch_obs:
            for source in (observation.raw_trace_path, observation.trace_view_path, observation.evidence_path):
                destination = trace_dir / source.name
                if destination.exists():
                    raise RuntimeError(f"Refusing to overwrite existing public trace: {destination}")
                shutil.copyfile(source, destination)
            save_observation(candidate, observation, trace_dir)
        observations.extend(batch_obs)
        if not reusable(candidate, task, trace_dir):
            raise RuntimeError(f"Trial was interrupted or empty; pausing after {task}")
    return observations


def append_lineage(rows: list[dict]) -> None:
    path = RUN / "branch_lineage.jsonl"
    existing = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    keys = {(x.get("batch"), x.get("round"), x.get("branch")): x for x in existing}
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            key = (row["batch"], row["round"], row["branch"])
            if key not in keys:
                handle.write(json.dumps(row) + "\n")
                keys[key] = row
            elif keys[key] != row:
                raise RuntimeError(f"Conflicting lineage checkpoint: {key}")


def append_score(batch: int, tasks: list[str], harness: str, observations: list[bridge.TrialObservation], *, recovered: bool = False) -> None:
    if len(observations) != len(tasks) or {x.task_name for x in observations} != set(tasks):
        raise RuntimeError("Cannot commit a partial batch score")
    correct, count, _ = score_observations(observations)
    row = {
        "batch": batch,
        "tasks": tasks,
        "harness": harness,
        "selected_correct": correct,
        "selected_total": count,
    }
    if recovered:
        row["score_recovered_by_selected_rerun"] = True
    existing = [json.loads(x) for x in (RUN / "opt_log.jsonl").read_text().splitlines() if x.strip()]
    if any(x["batch"] == batch for x in existing):
        raise RuntimeError(f"Batch {batch} is already committed; refusing duplicate score")
    with (RUN / "opt_log.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")
    print(f"   SCORE b{batch}: {correct}/{count}; committed {harness}", flush=True)


def finish_b8(tasks: list[str], incumbent: str) -> str:
    trace_dir = RUN / "traces" / "b8"
    r0 = [
        incumbent,
        "cand_full89_deepseek_tthe_continue_b6_b17_g2r2_t100_b8r0_g1",
    ]
    # b8r0 is already recorded in branch_lineage.  Preserve its fallback for g0.
    for candidate in dict.fromkeys(r0):
        if not loadable(candidate, PYTHON, MODEL, tasks):
            raise RuntimeError(f"b8 candidate is not loadable: {candidate}")
        observe(candidate, tasks, trace_dir)

    r1 = [
        "cand_full89_deepseek_tthe_continue_b6_b17_g2r2_t100_b8r1_g0",
        "cand_full89_deepseek_tthe_continue_b6_b17_g2r2_t100_b8r1_g1",
    ]
    lineage = []
    parents = [incumbent, r0[1]]
    active_r1 = []
    for branch, (parent, child) in enumerate(zip(parents, r1)):
        accepted = child if loadable(child, PYTHON, MODEL, tasks) else parent
        active_r1.append(accepted)
        lineage.append({
            "batch": 8,
            "round": 1,
            "branch": branch,
            "parent": parent,
            "proposed": child,
            "active": accepted,
            "fell_back": accepted == parent,
        })
        observe(accepted, tasks, trace_dir)
    append_lineage(lineage)

    candidates = list(dict.fromkeys([incumbent, *r0[1:], *active_r1]))
    print(f"   JUDGE b8 candidates={candidates}", flush=True)
    picked = proposer.pick_batch(
        candidates, trace_dir, RUN, "b8", PROPOSER_MODEL, 1800, incumbent
    )
    if picked not in candidates:
        raise RuntimeError(f"b8 judge did not select a valid candidate: {picked!r}")
    score_dir = RUN / "traces" / "b8_score"
    score_obs = bridge.observe_candidate(
        picked,
        f"agents.{picked}:AgentHarness",
        tasks,
        score_dir,
        dataset=DATASET,
        model=MODEL,
        environment="docker",
        concurrency=5,
        max_turns=100,
        harbor_command=HARBOR,
        timeout=14400,
        workdir=ROOT,
    )
    append_score(8, tasks, picked, score_obs, recovered=True)
    return picked


def run_batch(batch: int, tasks: list[str], incumbent: str) -> str:
    trace_dir = RUN / "traces" / f"b{batch}_safe_resume"
    observed: dict[str, list[bridge.TrialObservation]] = {}

    def ensure_observed(candidate: str) -> None:
        if candidate in observed:
            return
        observed[candidate] = observe(candidate, tasks, trace_dir)

    branches = [incumbent, incumbent]
    lineage = []
    for round_index in range(2):
        for candidate in dict.fromkeys(branches):
            if not loadable(candidate, PYTHON, MODEL, tasks):
                raise RuntimeError(f"candidate is not loadable: {candidate}")
            ensure_observed(candidate)
        tag = f"b{batch}r{round_index}"
        proposed = proposer.sample_branches(
            branches, trace_dir, RUN, tag, RUN.name.rsplit("_2026", 1)[0],
            PROPOSER_MODEL, 1800, reuse_existing=True,
        )
        next_branches = []
        for branch, (parent, child) in enumerate(zip(branches, proposed)):
            active = child if child and loadable(child, PYTHON, MODEL, tasks) else parent
            next_branches.append(active)
            lineage.append({
                "batch": batch, "round": round_index, "branch": branch,
                "parent": parent, "proposed": child, "active": active,
                "fell_back": active == parent,
            })
        branches = next_branches
        for candidate in dict.fromkeys(branches):
            ensure_observed(candidate)
    append_lineage(lineage)
    candidates = list(observed)
    print(f"   JUDGE b{batch} candidates={candidates}", flush=True)
    picked = choose_checkpoint(candidates, trace_dir, batch, incumbent)
    if picked not in candidates:
        raise RuntimeError(f"b{batch} judge did not select a valid candidate: {picked!r}")
    score_obs = observed[picked]
    if len(score_obs) != len(tasks):
        score_obs = observe(picked, tasks, RUN / "traces" / f"b{batch}_safe_score")
    append_score(batch, tasks, picked, score_obs)
    return picked


def choose_checkpoint(candidates: list[str], trace_dir: Path, batch: int, incumbent: str) -> str | None:
    choice_path = RUN / f"judge_b{batch}.txt"
    if choice_path.exists():
        picked = choice_path.read_text().strip()
        if picked not in candidates:
            raise RuntimeError("Cached judge selected an unknown candidate")
        return picked
    return proposer.pick_batch(candidates, trace_dir, RUN, f"b{batch}", PROPOSER_MODEL, 1800, incumbent)


def resume_partial_b10(tasks: list[str], incumbent: str) -> str:
    """Complete b10 from the two r0 candidates already observed before pausing."""
    trace_dir = RUN / "traces" / "b10_safe_resume"
    trace_dir.mkdir(parents=True, exist_ok=True)
    old_trace_dir = RUN / "traces" / "b10"
    r0 = [
        "cand_full89_deepseek_tthe_continue_b6_b17_g2r2_t100_b10r0_g0",
        "cand_full89_deepseek_tthe_continue_b6_b17_g2r2_t100_b10r0_g1",
    ]
    observed = {}
    for candidate in [incumbent, *r0]:
        if not loadable(candidate, PYTHON, MODEL, tasks):
            raise RuntimeError(f"b10 checkpoint candidate is not loadable: {candidate}")
        for task in tasks:
            if reusable(candidate, task, old_trace_dir) and not reusable(candidate, task, trace_dir):
                for source in old_trace_dir.glob(f"{candidate}__{task}__a0.*"):
                    destination = trace_dir / source.name
                    if not destination.exists():
                        shutil.copyfile(source, destination)
        observed[candidate] = observe(candidate, tasks, trace_dir)

    lineage = []
    for branch, child in enumerate(r0):
        lineage.append({
            "batch": 10, "round": 0, "branch": branch,
            "parent": incumbent, "proposed": child, "active": child,
            "fell_back": False,
        })

    append_lineage(lineage)
    proposed = proposer.sample_branches(
        r0, trace_dir, RUN, "b10r1", RUN.name.rsplit("_2026", 1)[0],
        PROPOSER_MODEL, 1800, reuse_existing=True,
    )
    active_r1 = []
    for branch, (parent, child) in enumerate(zip(r0, proposed)):
        active = child if child and loadable(child, PYTHON, MODEL, tasks) else parent
        active_r1.append(active)
        lineage.append({
            "batch": 10, "round": 1, "branch": branch,
            "parent": parent, "proposed": child, "active": active,
            "fell_back": active == parent,
        })
        observed[active] = observe(active, tasks, trace_dir)
    append_lineage(lineage)

    candidates = list(dict.fromkeys([incumbent, *r0, *active_r1]))
    print(f"   JUDGE b10 checkpoint candidates={candidates}", flush=True)
    picked = choose_checkpoint(candidates, trace_dir, 10, incumbent)
    if picked not in candidates:
        raise RuntimeError(f"b10 checkpoint judge did not select a valid candidate: {picked!r}")
    score_obs = observed[picked]
    recovered = len(score_obs) != len(tasks)
    if recovered:
        score_obs = observe(picked, tasks, RUN / "traces" / "b10_safe_score")
    append_score(10, tasks, picked, score_obs, recovered=recovered)
    return picked


def main() -> None:
    clear_proxy()
    PRIVATE.mkdir(parents=True, exist_ok=True, mode=0o700)
    # Keep the lock open through all observations/proposer/judge work.
    lock = (PRIVATE / "resume.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    config = json.loads((RUN / "config.json").read_text(encoding="utf-8"))
    tasks = list(config["tasks"])
    if os.environ.get("TTHE_RESUME_FROM_BATCH") == "10":
        if (config["model"] != MODEL or config["group"] != 2 or config["max_rounds"] != 2
                or config["max_turns"] != 100 or config["proposer_model"] != PROPOSER_MODEL):
            raise RuntimeError("Frozen experiment configuration does not match this continuation")
        if not os.environ.get("OPENAI_API_KEY") or not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("Original solver/proposer credentials are not configured")
        committed = {x["batch"]: x for x in map(json.loads, (RUN / "opt_log.jsonl").read_text().splitlines())}
        if 11 in committed:
            print("DONE: b10 and b11 are already committed", flush=True)
            return
        if 10 in committed:
            if pause_after(10):
                return
            preflight_docker()
            run_batch(11, tasks[55:60], committed[10]["harness"])
            return
        preflight_docker()
        incumbent = (RUN / "judge_b9.txt").read_text(encoding="utf-8").strip()
        if not loadable(incumbent, PYTHON, MODEL, tasks):
            raise RuntimeError(f"b9 incumbent is not loadable: {incumbent}")
        incumbent = resume_partial_b10(tasks[50:55], incumbent)
        if pause_after(10):
            return
        incumbent = run_batch(11, tasks[55:60], incumbent)
        print(f"DONE continuation through b11; final harness={incumbent}", flush=True)
        return
    if (RUN / "judge_b8.txt").exists() or any('"batch": 8' in x for x in (RUN / "opt_log.jsonl").read_text().splitlines()):
        raise SystemExit("b8 is already committed; refusing duplicate continuation")
    preflight_docker()
    incumbent = (RUN / "judge_b7.txt").read_text(encoding="utf-8").strip()
    if not loadable(incumbent, PYTHON, MODEL, tasks):
        raise RuntimeError(f"b7 incumbent is not loadable: {incumbent}")
    print(f"RESUME Terminal-Bench from b8; incumbent={incumbent}", flush=True)
    incumbent = finish_b8(tasks[40:45], incumbent)
    if pause_after(8):
        return
    for batch, start in ((9, 45), (10, 50), (11, 55)):
        incumbent = run_batch(batch, tasks[start : start + 5], incumbent)
        if pause_after(batch):
            return
    print(f"DONE continuation through b11; final harness={incumbent}", flush=True)


if __name__ == "__main__":
    def interrupted(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    try:
        main()
    finally:
        remove_idle_network()
