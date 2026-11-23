"""Trace-driven Test-Time Harness Evolution for Terminal-Bench 2.0.

This is the Terminal-Bench analogue of the formal TTHE domain loops:

  OBSERVE  -> run each active AgentHarness once per task and save agent-only traces
  GENERATE -> advance G fixed proposer branches for R rounds
  PICK     -> a label-free judge commits one observed harness, with rollback to H
  SCORE    -> reveal only that committed harness's sealed Harbor rewards

Harbor computes its verifier result in the same trial as the agent.  The bridge
keeps those results only in controller memory; proposer-visible files contain
the byte-exact agent trajectory and label-free evidence, never verifier data.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import time
from pathlib import Path

from dotenv import load_dotenv

import tthe_bridge as bridge
import tthe_proposer as proposer
from audit_agent import audit_file


ROOT = Path(__file__).resolve().parent
AGENTS_DIR = ROOT / "agents"
DEFAULT_TASKS = ROOT / "slices" / "smoke.json"


def load_tasks(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    values = data.get("tasks") if isinstance(data, dict) else data
    if not isinstance(values, list) or not values:
        raise ValueError(
            f'{path} must contain a non-empty JSON list or {{"tasks": [...]}}'
        )
    tasks = [str(value) for value in values]
    if len(tasks) != len(set(tasks)):
        raise ValueError(f"duplicate task names in {path}")
    return tasks


def import_path(candidate: str) -> str:
    return f"agents.{candidate}:AgentHarness"


def loadable(candidate: str, python_command: str, model: str, tasks: list[str]) -> bool:
    path = AGENTS_DIR / f"{candidate}.py"
    if not path.exists():
        return False
    violations = audit_file(path, tuple(tasks))
    if violations:
        print(
            f"   [audit] REJECTED {candidate}: "
            + "; ".join(
                f"{item['rule']} line {item['line']}: {item['detail']}"
                for item in violations[:5]
            ),
            flush=True,
        )
        return False
    module = import_path(candidate).split(":", 1)[0]
    command = shlex.split(python_command) + [
        "-c",
        f"from {module} import AgentHarness; print(AgentHarness.__name__)",
    ]
    try:
        result = subprocess.run(
            command,
            cwd=str(ROOT),
            env=bridge.direct_env(model),
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    if result.returncode != 0:
        print(
            f"   [import] REJECTED {candidate}: {(result.stderr or result.stdout)[-500:]}",
            flush=True,
        )
        return False
    return True


def score_observations(
    observations: list[bridge.TrialObservation],
) -> tuple[int, int, list[dict]]:
    rows = []
    for observation in observations:
        reward = observation.reward if observation.reward is not None else 0.0
        rows.append(
            {
                "task": observation.task_name,
                "attempt": observation.attempt,
                "reward": reward,
                "resolved": reward > 0,
            }
        )
    return sum(row["resolved"] for row in rows), len(rows), rows


def main() -> None:
    load_dotenv(ROOT / ".env", override=True)
    parser = argparse.ArgumentParser(description="TTHE on Terminal-Bench 2.0")
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--dataset", default="terminal-bench@2.0")
    parser.add_argument("--model", default="anthropic/claude-opus-4-6")
    parser.add_argument("--environment", default="runloop")
    parser.add_argument("--initial-harness", default="baseline_kira")
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--group", type=int, default=2)
    parser.add_argument("--max-rounds", type=int, default=2)
    parser.add_argument("--concurrent", type=int, default=5)
    parser.add_argument("--max-turns", type=int, default=100)
    parser.add_argument("--proposer-model", default="opus")
    parser.add_argument("--propose-timeout", type=int, default=1800)
    parser.add_argument("--harbor-timeout", type=int, default=14400)
    parser.add_argument("--harbor-command", default="uv run harbor")
    parser.add_argument("--python-command", default="uv run python")
    parser.add_argument("--run-name", default="tb2_tthe")
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate task spec, seed harness, and commands without running Harbor or a proposer",
    )
    args = parser.parse_args()

    if (
        args.batch_size < 1
        or args.group < 1
        or args.max_rounds < 1
        or args.concurrent < 1
        or args.max_turns < 1
    ):
        parser.error(
            "batch-size, group, max-rounds, concurrent, and max-turns must all be positive"
        )

    tasks = load_tasks(args.tasks)
    if not loadable(args.initial_harness, args.python_command, args.model, tasks):
        raise SystemExit(f"initial harness is not loadable: {args.initial_harness}")

    preview = bridge.build_harbor_command(
        args.harbor_command,
        import_path(args.initial_harness),
        "observe-preview",
        Path("<temporary-controller-only-dir>"),
        tasks[: args.batch_size],
        dataset=args.dataset,
        model=args.model,
        environment=args.environment,
        concurrency=min(args.concurrent, len(tasks[: args.batch_size])),
        max_turns=args.max_turns,
    )
    if args.validate_only:
        print(f"validated {len(tasks)} tasks and harness {args.initial_harness}")
        print("observe command:", shlex.join(preview))
        return

    if args.fresh:
        for path in AGENTS_DIR.glob("cand_*.py"):
            path.unlink()

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = ROOT / "logs" / f"{args.run_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    batches = [
        tasks[index : index + args.batch_size]
        for index in range(0, len(tasks), args.batch_size)
    ]
    config = {
        "dataset": args.dataset,
        "tasks_file": str(args.tasks.resolve()),
        "tasks": tasks,
        "model": args.model,
        "environment": args.environment,
        "initial_harness": args.initial_harness,
        "batch_size": args.batch_size,
        "group": args.group,
        "max_rounds": args.max_rounds,
        "concurrent": args.concurrent,
        "max_turns": args.max_turns,
        "proposer_model": args.proposer_model,
        "trials_per_candidate_task": 1,
        "gold_visibility": "sealed in controller memory until judge commit",
        "proxy_aggregation": None,
    }
    (run_dir / "config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    log_path = run_dir / "opt_log.jsonl"
    branch_path = run_dir / "branch_lineage.jsonl"

    incumbent = args.initial_harness
    total_correct = 0
    total_trials = 0
    selected_rows: list[dict] = []
    print(
        f"TTHE Terminal-Bench 2: {len(tasks)} tasks, {len(batches)} batches, "
        f"G={args.group}, R={args.max_rounds}, initial={incumbent}",
        flush=True,
    )

    with (
        log_path.open("w", encoding="utf-8") as opt_log,
        branch_path.open("w", encoding="utf-8") as branch_log,
    ):
        for batch_index, batch in enumerate(batches):
            trace_dir = run_dir / "traces" / f"b{batch_index}"
            trace_dir.mkdir(parents=True)
            observed: dict[str, list[bridge.TrialObservation]] = {}
            branches = [incumbent] * args.group
            print(
                f"\n===== BATCH {batch_index}/{len(batches)} tasks={batch} H={incumbent} =====",
                flush=True,
            )

            def observe(candidate: str) -> None:
                if candidate in observed:
                    return
                print(f"   OBSERVE {candidate}", flush=True)
                observed[candidate] = bridge.observe_candidate(
                    candidate,
                    import_path(candidate),
                    batch,
                    trace_dir,
                    dataset=args.dataset,
                    model=args.model,
                    environment=args.environment,
                    concurrency=args.concurrent,
                    max_turns=args.max_turns,
                    harbor_command=args.harbor_command,
                    timeout=args.harbor_timeout,
                    workdir=ROOT,
                )

            for round_index in range(args.max_rounds):
                for candidate in dict.fromkeys(branches):
                    if loadable(candidate, args.python_command, args.model, tasks):
                        observe(candidate)

                tag = f"b{batch_index}r{round_index}"
                proposed = proposer.sample_branches(
                    branches,
                    trace_dir,
                    run_dir,
                    tag,
                    args.run_name,
                    args.proposer_model,
                    args.propose_timeout,
                )
                next_branches = []
                lineage = []
                for branch_index, (parent, child) in enumerate(zip(branches, proposed)):
                    accepted = (
                        child
                        if child
                        and loadable(child, args.python_command, args.model, tasks)
                        else parent
                    )
                    next_branches.append(accepted)
                    lineage.append(
                        {
                            "batch": batch_index,
                            "round": round_index,
                            "branch": branch_index,
                            "parent": parent,
                            "proposed": child,
                            "active": accepted,
                            "fell_back": accepted == parent,
                        }
                    )
                branches = next_branches
                for candidate in dict.fromkeys(branches):
                    observe(candidate)
                for row in lineage:
                    branch_log.write(json.dumps(row) + "\n")
                branch_log.flush()
                print(
                    f"   GENERATE round {round_index}: "
                    f"{sum(not row['fell_back'] for row in lineage)}/{len(lineage)} advanced",
                    flush=True,
                )

            candidates = list(observed)
            picked = proposer.pick_batch(
                candidates,
                trace_dir,
                run_dir,
                f"b{batch_index}",
                args.proposer_model,
                args.propose_timeout,
                incumbent,
            )
            incumbent = picked if picked in candidates else incumbent
            correct, count, rows = score_observations(observed[incumbent])
            total_correct += correct
            total_trials += count
            for row in rows:
                row.update({"batch": batch_index, "harness": incumbent})
                selected_rows.append(row)
            record = {
                "batch": batch_index,
                "tasks": batch,
                "harness": incumbent,
                "branches": branches,
                "candidates": candidates,
                "selected_correct": correct,
                "selected_total": count,
            }
            opt_log.write(json.dumps(record) + "\n")
            opt_log.flush()
            print(
                f"   PICK {incumbent}; post-commit Harbor reward = {correct}/{count}",
                flush=True,
            )

            # Discard every unselected reward before the next batch. Only selected rewards were serialized.
            observed.clear()

    result = {
        "correct": total_correct,
        "total": total_trials,
        "accuracy": total_correct / total_trials if total_trials else 0.0,
        "final_harness": incumbent,
        "selected_trials": selected_rows,
    }
    (run_dir / "result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\nRESULT {total_correct}/{total_trials}; final harness={incumbent}")
    print(f"saved to {run_dir}")


if __name__ == "__main__":
    main()
