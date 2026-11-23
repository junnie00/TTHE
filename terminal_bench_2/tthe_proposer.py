"""Agentic fixed-branch proposer and judge for Terminal-Bench 2 TTHE."""

from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import claude_wrapper


ROOT = Path(__file__).resolve().parent
AGENTS_DIR = ROOT / "agents"
PROPOSER_TOOLS = ["Read", "Glob", "Grep", "Write", "Edit"]
PROXY_ENV_VARS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)

DIVERSITY = [
    (
        "CONSERVATIVE REPAIR ROLE: preserve every behavior supported by trace evidence. Make the smallest "
        "general change that fixes one reproducible harness failure; a no-op is preferable to speculation."
    ),
    (
        "INDEPENDENT EXPLORATION ROLE: test a different causal hypothesis about why the parent wastes turns, "
        "misreads observations, fails to verify, or stops incorrectly. Implement the simplest surviving mechanism."
    ),
    (
        "ADVERSARIAL AUDIT ROLE: start from disagreements among candidates. Look for counterexamples in actual "
        "commands and observations, then retain only mechanisms supported across tasks."
    ),
]


def _safe(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_") or "run"


def propose_branch(task: dict) -> bool:
    candidates = task["candidates"]
    base = task["base_candidate"]
    name = task["new_name"]
    branch_id = int(task["branch_id"])
    source = AGENTS_DIR / f"{base}.py"
    target = AGENTS_DIR / f"{name}.py"
    if base not in candidates or not source.exists():
        return False
    target.unlink(missing_ok=True)
    shutil.copyfile(source, target)

    rows = []
    for candidate in candidates:
        relation = "ASSIGNED BASE" if candidate == base else "PEER EVIDENCE"
        rows.append(
            f"- `{candidate}` [{relation}] source `{AGENTS_DIR / f'{candidate}.py'}`; "
            f"traces `{task['trace_dir']}/{candidate}__*`"
        )
    prompt = f"""You are evolving a GENERAL Terminal-Bench 2 executable agent harness around a FROZEN model.

The harness is arbitrary Python compatible with Harbor's custom-agent interface. The model identity,
endpoint, credentials, and maximum episode budget are fixed. Gold verifier tests and rewards are unavailable.

CANDIDATES
{chr(10).join(rows)}

BRANCH LINEAGE
- You are branch G{branch_id}; your assigned parent is `{base}`.
- `{target}` is already a byte-for-byte copy of that parent. Edit only this file.
- Peer implementations and traces are evidence, not alternate parents. Do not replace the file wholesale
  with a peer. If evidence is insufficient, leave the copied child unchanged.

EVIDENCE CONTRACT
For every candidate/task there can be three files:
- `*.raw_trajectory.json`: byte-exact, untruncated ATIF agent trajectory;
- `*.md`: bounded reading view;
- `*.evidence.json`: factual command/observation index and operational metadata.
The evidence index is NOT a reward and contains no correctness score. Deep-read the raw trajectories when the
view elides relevant material. Diagnose causal behavior: which instructions were misunderstood, which command
or observation caused the trajectory to go off course, whether the agent built and ran a task-grounded check,
whether its claimed completion followed from observable evidence, and where turns were wasted. A non-empty
command history or a task-complete call is not proof of success.

ACTION SPACE
You may change the executable loop, prompt construction, tool schema, command parsing/execution, context
management, verification and stopping policy. Do not change `self._model_name`, `self._llm`, or
`self._max_episodes`; do not provide another model/API endpoint; do not access Harbor verifier artifacts,
tests, solutions, rewards, or task-specific answer knowledge. Never hardcode task names or special cases.
Keep the class name `AgentHarness` and Harbor compatibility.

Use a reproducible failure visible in the parent/peer traces to justify every behavioral change. Preserve
behaviors that already work. You cannot execute Harbor or the child from this session; the controller will
audit and run it exactly once per task after you exit.

SEARCH STANCE
{task["role"]}

Edit ONLY `{target}` and then stop. Do not create reports, result files, prompts, or additional modules.
"""
    result = claude_wrapper.run(
        prompt=prompt,
        model=task["model"],
        allowed_tools=PROPOSER_TOOLS,
        cwd=str(ROOT),
        log_dir=str(Path(task["run_dir"]) / "claude_sessions"),
        name=f"{task['tag']}_g{branch_id}",
        timeout_seconds=int(task["timeout"]),
        progress=False,
    )
    if result.exit_code != 0:
        target.unlink(missing_ok=True)
        return False
    return target.exists()


def _prepare_judge_provenance(task: dict) -> Path:
    """Write exact controller-known ancestry and immediate-parent source diffs."""
    run_dir = Path(task["run_dir"])
    tag = _safe(task["tag"])
    lineage_path = Path(task.get("lineage_path", run_dir / "branch_lineage.jsonl"))
    lineage = []
    if lineage_path.exists():
        for line in lineage_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                lineage.append(json.loads(line))

    evidence_dir = run_dir / f"judge_{tag}.provenance"
    evidence_dir.mkdir(exist_ok=True)
    candidate_records = []
    for candidate in task["candidates"]:
        records = [
            row
            for row in lineage
            if row.get("proposed") == candidate
            or (
                row.get("active") == candidate
                and row.get("parent") != candidate
                and not row.get("fell_back", False)
            )
        ]
        relation = records[-1] if records else None
        parent = relation.get("parent") if relation else None
        diff_path = None
        if parent:
            parent_path = AGENTS_DIR / f"{parent}.py"
            candidate_path = AGENTS_DIR / f"{candidate}.py"
            if parent_path.exists() and candidate_path.exists():
                diff_path = evidence_dir / f"{_safe(candidate)}.diff"
                diff = difflib.unified_diff(
                    parent_path.read_text(encoding="utf-8").splitlines(keepends=True),
                    candidate_path.read_text(encoding="utf-8").splitlines(keepends=True),
                    fromfile=str(parent_path),
                    tofile=str(candidate_path),
                )
                diff_path.write_text("".join(diff), encoding="utf-8")
        candidate_records.append(
            {
                "candidate": candidate,
                "role": (
                    "incumbent" if candidate == task["incumbent"] else "challenger"
                ),
                "source_path": str(AGENTS_DIR / f"{candidate}.py"),
                "immediate_parent": parent,
                "lineage_records": records,
                "immediate_parent_diff_path": str(diff_path) if diff_path else None,
            }
        )

    provenance_path = run_dir / f"judge_{tag}.provenance.json"
    provenance_path.write_text(
        json.dumps(
            {
                "incumbent": task["incumbent"],
                "candidates": candidate_records,
                "full_lineage": lineage,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return provenance_path


def _judge_prompt(task: dict, provenance_path: Path) -> str:
    candidates = task["candidates"]
    incumbent = task["incumbent"]
    choice_path = Path(task["choice_path"])
    matrix_path = choice_path.with_suffix(".matrix.md")
    rows = "\n".join(
        f"- `{candidate}`: source `{AGENTS_DIR / f'{candidate}.py'}`; "
        f"traces `{task['trace_dir']}/{candidate}__*`"
        for candidate in candidates
    )
    return f"""You are the label-free JUDGE for Terminal-Bench 2 harness evolution.

Choose exactly one GENERAL harness that is most credibly effective across this batch. You may choose any
listed candidate, including the incumbent. You cannot see Harbor verifier tests or rewards and must not write
or modify harness code. The controller has not filtered or ranked candidates for you.

CANDIDATES
{rows}

CONTROLLER-KNOWN PROVENANCE
Read `{provenance_path}` before comparing candidates. It contains mechanically recorded parent/lineage facts
and paths to exact unified diffs against each candidate's immediate parent. Treat those records as factual
inputs, not as recommendations. Inspect candidate sources when a diff or provenance record is unavailable.
Do not infer ancestry from source similarity, candidate names, or apparent feature overlap.

Each task has a byte-exact `*.raw_trajectory.json`, a bounded `*.md` view, and a non-scalar
`*.evidence.json` index. Inspect the raw trace whenever the view is insufficient. Judge from observable,
task-grounded evidence: whether the agent understood the requested end state, used terminal observations
correctly, constructed meaningful checks, reacted to failures, and had evidence for completion. Do not reward
mere activity, token use, a task-complete call, confident prose, or architectural elegance. Exceptions,
task-complete counts, and other operational metadata are contextual evidence, never automatic eligibility or
disqualification rules.

Perform one evidence-grounded review in two explicit stages:

STAGE 1 — EVIDENCE MATRIX
1. Compare candidates mainly on tasks where their behavior differs. Agreement is a weak prior, not proof.
2. An agent-authored check is evidence only to the extent that it actually tests the task contract.
3. Missing trace material is absence of evidence, not proof of failure.
4. Write `{matrix_path}` with one row per candidate and one column per task. Each cell must be
   CREDIBLE / NOT-CREDIBLE / UNRESOLVED followed by a short pointer to concrete trace evidence.

STAGE 2 — DECISION SELF-CHECK
5. Add a provenance section to the matrix stating each candidate's mechanically recorded immediate parent and
   summarizing only changes visible in its exact parent diff. Do not describe one parallel branch as containing
   another branch's changes unless the provenance records say so.
6. Name a provisional winner, then actively seek the strongest trace-grounded counterevidence against it and
   the strongest evidence for `{incumbent}`. Revisit any exception or missing completion in its trace context;
   neither fact decides the outcome by itself.
7. Reconcile the final decision against the complete matrix. The burden of proof is on challengers; genuine
   ties go to the incumbent, but a challenger may win whenever the evidence supports it.
8. Never infer or search for verifier outputs. They are deliberately not present.

No task may be omitted. After both stages, write ONLY the chosen exact candidate name to `{choice_path}` using
the Write tool and stop.
"""


def pick_candidate(task: dict) -> bool:
    choice_path = Path(task["choice_path"])
    provenance_path = _prepare_judge_provenance(task)
    prompt = _judge_prompt(task, provenance_path)
    result = claude_wrapper.run(
        prompt=prompt,
        model=task["model"],
        allowed_tools=PROPOSER_TOOLS,
        cwd=str(ROOT),
        log_dir=str(Path(task["run_dir"]) / "claude_sessions"),
        name=f"judge_{task['tag']}",
        timeout_seconds=int(task["timeout"]),
        progress=False,
    )
    return result.exit_code == 0 and choice_path.exists()


def _wait_and_kill(processes: list[subprocess.Popen], timeout: int) -> None:
    deadline = time.time() + timeout + 120
    try:
        for process in processes:
            try:
                process.wait(timeout=max(1, deadline - time.time()))
            except subprocess.TimeoutExpired:
                pass
    finally:
        for process in processes:
            if process.poll() is None:
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass


def sample_branches(
    branches: list[str],
    trace_dir: Path,
    run_dir: Path,
    tag: str,
    run_name: str,
    model: str,
    timeout: int,
    *,
    reuse_existing: bool = False,
) -> list[str | None]:
    active = list(dict.fromkeys(branches))
    names = [
        f"cand_{_safe(run_name)}_{_safe(tag)}_g{index}"
        for index in range(len(branches))
    ]
    processes = []
    for index, (base, name) in enumerate(zip(branches, names)):
        task_path = run_dir / f"propose_{_safe(tag)}_g{index}.json"
        if reuse_existing and task_path.exists() and (AGENTS_DIR / f"{name}.py").exists():
            previous = json.loads(task_path.read_text(encoding="utf-8"))
            if previous["base_candidate"] != base or previous["new_name"] != name:
                raise RuntimeError(f"Proposal checkpoint parent/name mismatch: {task_path}")
            continue
        (AGENTS_DIR / f"{name}.py").unlink(missing_ok=True)
        task = {
            "candidates": active,
            "base_candidate": base,
            "new_name": name,
            "branch_id": index,
            "trace_dir": str(trace_dir),
            "run_dir": str(run_dir),
            "tag": tag,
            "model": model,
            "timeout": timeout,
            "role": DIVERSITY[index % len(DIVERSITY)],
        }
        task_path.write_text(json.dumps(task, indent=2), encoding="utf-8")
        processes.append(
            subprocess.Popen(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--worker",
                    str(task_path),
                ],
                cwd=str(ROOT),
                start_new_session=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        )
    _wait_and_kill(processes, timeout)
    return [name if (AGENTS_DIR / f"{name}.py").exists() else None for name in names]


def pick_batch(
    candidates: list[str],
    trace_dir: Path,
    run_dir: Path,
    tag: str,
    model: str,
    timeout: int,
    incumbent: str,
) -> str | None:
    if len(candidates) <= 1:
        return candidates[0] if candidates else None
    choice_path = run_dir / f"judge_{_safe(tag)}.txt"
    choice_path.unlink(missing_ok=True)
    task = {
        "candidates": candidates,
        "incumbent": incumbent,
        "trace_dir": str(trace_dir),
        "run_dir": str(run_dir),
        "tag": tag,
        "model": model,
        "timeout": timeout,
        "choice_path": str(choice_path),
        "lineage_path": str(run_dir / "branch_lineage.jsonl"),
    }
    task_path = run_dir / f"judge_{_safe(tag)}.json"
    task_path.write_text(json.dumps(task, indent=2), encoding="utf-8")
    process = subprocess.Popen(
        [sys.executable, str(Path(__file__).resolve()), "--picker", str(task_path)],
        cwd=str(ROOT),
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    _wait_and_kill([process], timeout)
    if choice_path.exists():
        chosen = choice_path.read_text(encoding="utf-8").strip().split()
        if chosen and chosen[0] in candidates:
            return chosen[0]
    return None


def main() -> None:
    for key in PROXY_ENV_VARS:
        os.environ.pop(key, None)
    os.environ["NO_PROXY"] = "*"
    os.environ["no_proxy"] = "*"
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--picker", type=Path)
    args = parser.parse_args()
    task_path = args.worker or args.picker
    if task_path is None:
        parser.error("one of --worker or --picker is required")
    task = json.loads(task_path.read_text(encoding="utf-8"))
    ok = propose_branch(task) if args.worker else pick_candidate(task)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
