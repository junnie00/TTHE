"""Harbor bridge for label-free TTHE on Terminal-Bench 2.0.

Harbor runs the agent and verifier in one trial.  This bridge treats the job
directory as a sealed controller-only staging area: it copies only the agent
trajectory into the public run directory, returns verifier rewards to the
controller in memory, and then lets the temporary Harbor directory disappear.
No verifier result or verifier log is written into a proposer-visible trace.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROXY_ENV_VARS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)


@dataclass
class TrialObservation:
    """One Harbor trial.  ``reward`` is controller-private until commit."""

    task_name: str
    attempt: int
    reward: float | None
    status: dict[str, Any]
    raw_trace_path: Path
    trace_view_path: Path
    evidence_path: Path


def direct_env(model: str) -> dict[str, str]:
    """Return a subprocess environment that never inherits proxy settings."""
    env = os.environ.copy()
    for key in PROXY_ENV_VARS:
        env.pop(key, None)
    env["NO_PROXY"] = "*"
    env["no_proxy"] = "*"
    env["HARBOR_MODEL"] = model
    return env


def build_harbor_command(
    harbor_command: str,
    import_path: str,
    job_name: str,
    jobs_dir: Path,
    tasks: list[str],
    *,
    dataset: str,
    model: str,
    environment: str,
    concurrency: int,
    max_turns: int,
) -> list[str]:
    """Build the exact one-attempt Harbor command used by OBSERVE."""
    cmd = shlex.split(harbor_command) + [
        "run",
        "--agent-import-path",
        import_path,
        "-d",
        dataset,
        "-m",
        model,
        "-e",
        environment,
        "-n",
        str(concurrency),
        "--n-attempts",
        "1",
        "--yes",
        "--agent-setup-timeout-multiplier",
        "5",
        "--agent-kwarg",
        "record_terminal_session=false",
        "--agent-kwarg",
        f"max_turns={max_turns}",
        "--job-name",
        job_name,
        "--jobs-dir",
        str(jobs_dir),
    ]
    for task in tasks:
        cmd.extend(["-i", task])
    if environment == "docker" and os.environ.get("TTHE_SAFE_DOCKER") == "1":
        cmd.extend(["--environment-import-path", "safe_docker:SafeDockerEnvironment"])
    return cmd


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "item"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _reward(result: dict[str, Any]) -> float | None:
    verifier = result.get("verifier_result") or {}
    rewards = verifier.get("rewards") or {}
    value = rewards.get("reward")
    if value is None and len(rewards) == 1:
        value = next(iter(rewards.values()))
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _content(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, indent=2)


def _bounded(value: Any, limit: int = 8000) -> str:
    """Bound only the Markdown view; raw_trajectory.json stays byte-exact."""
    text = _content(value)
    if len(text) <= limit:
        return text
    head = limit // 2
    tail = limit - head
    return f"{text[:head]}\n\n... [{len(text) - limit} chars elided in view] ...\n\n{text[-tail:]}"


def _trajectory_facts(trajectory: dict[str, Any]) -> dict[str, Any]:
    steps = trajectory.get("steps") or []
    commands: list[dict[str, Any]] = []
    completion_calls = 0
    for step in steps:
        if not isinstance(step, dict):
            continue
        observations = (step.get("observation") or {}).get("results") or []
        observation_text = [
            _content(item.get("content", ""))
            for item in observations
            if isinstance(item, dict)
        ]
        for call in step.get("tool_calls") or []:
            if not isinstance(call, dict):
                continue
            function = str(call.get("function_name", ""))
            if function == "mark_task_complete":
                completion_calls += 1
            commands.append(
                {
                    "step_id": step.get("step_id"),
                    "function": function,
                    "arguments": call.get("arguments") or {},
                    "observations": observation_text,
                }
            )
    return {
        "step_count": len(steps),
        "tool_call_count": len(commands),
        "task_complete_calls": completion_calls,
        "commands_and_observations": commands,
        "final_metrics": trajectory.get("final_metrics") or {},
    }


def _trace_markdown(
    candidate: str,
    task_name: str,
    attempt: int,
    trajectory: dict[str, Any],
    status: dict[str, Any],
    raw_name: str,
    evidence_name: str,
) -> str:
    lines = [
        f"# Trace — harness `{candidate}` — task `{task_name}` — attempt {attempt}",
        "",
        "The byte-exact, untruncated agent trajectory is stored in "
        f"`{raw_name}`. This Markdown file is a bounded reading view; `{evidence_name}` "
        "is a factual index, not a reward or correctness score.",
        "",
        "## RUN STATUS (label-free)",
        "",
        "```json",
        json.dumps(status, ensure_ascii=False, indent=2),
        "```",
        "",
        "## AGENT TRAJECTORY",
    ]
    for step in trajectory.get("steps") or []:
        if not isinstance(step, dict):
            continue
        sid = step.get("step_id", "?")
        source = step.get("source", "unknown")
        lines.extend(
            ["", f"### step {sid} — {source}", "", _bounded(step.get("message", ""))]
        )
        if step.get("reasoning_content"):
            lines.extend(["", "REASONING:", _bounded(step["reasoning_content"])])
        for call in step.get("tool_calls") or []:
            if not isinstance(call, dict):
                continue
            lines.extend(
                [
                    "",
                    f"TOOL `{call.get('function_name', '')}`:",
                    "```json",
                    _bounded(call.get("arguments") or {}),
                    "```",
                ]
            )
        for obs in (step.get("observation") or {}).get("results") or []:
            if isinstance(obs, dict):
                lines.extend(["", "OBSERVATION:", _bounded(obs.get("content", ""))])
    lines.extend(
        [
            "",
            "## INTERPRETATION RULE",
            "",
            "Commands, outputs, self-authored checks, and completion behavior are label-free evidence. "
            "They are not proof of correctness. No Harbor verifier output or reward appears in this trace.",
            "",
        ]
    )
    return "\n".join(lines)


def collect_harbor_observations(
    job_dir: Path,
    candidate: str,
    expected_tasks: list[str],
    trace_dir: Path,
) -> list[TrialObservation]:
    """Extract agent-only traces while keeping verifier data out of ``trace_dir``."""
    trace_dir.mkdir(parents=True, exist_ok=True)
    by_task: dict[str, list[Path]] = {}
    if job_dir.exists():
        for child in sorted(job_dir.iterdir()):
            if child.is_dir() and (
                (child / "result.json").exists()
                or (child / "agent" / "trajectory.json").exists()
            ):
                result = _read_json(child / "result.json")
                task_name = str(
                    result.get("task_name") or child.name.rsplit("__", 1)[0]
                )
                by_task.setdefault(task_name, []).append(child)

    observations: list[TrialObservation] = []
    for task_name in expected_tasks:
        trials = by_task.get(task_name, [])
        if not trials:
            stem = f"{_safe_name(candidate)}__{_safe_name(task_name)}__a0"
            raw_path = trace_dir / f"{stem}.raw_trajectory.json"
            evidence_path = trace_dir / f"{stem}.evidence.json"
            view_path = trace_dir / f"{stem}.md"
            raw_path.write_text("{}\n", encoding="utf-8")
            status = {
                "exception_type": "MissingTrial",
                "exception_message": "Harbor produced no trial",
            }
            evidence_path.write_text(
                json.dumps(
                    {
                        "candidate": candidate,
                        "task_name": task_name,
                        "attempt": 0,
                        "status": status,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            view_path.write_text(
                _trace_markdown(
                    candidate,
                    task_name,
                    0,
                    {},
                    status,
                    raw_path.name,
                    evidence_path.name,
                ),
                encoding="utf-8",
            )
            observations.append(
                TrialObservation(
                    task_name, 0, None, status, raw_path, view_path, evidence_path
                )
            )
            continue

        for attempt, trial_dir in enumerate(trials):
            result = _read_json(trial_dir / "result.json")
            trajectory_source = trial_dir / "agent" / "trajectory.json"
            trajectory = _read_json(trajectory_source)
            exception = result.get("exception_info") or {}
            agent_result = result.get("agent_result") or {}
            status = {
                "exception_type": exception.get("exception_type"),
                "exception_message": exception.get("exception_message"),
                "n_input_tokens": agent_result.get("n_input_tokens"),
                "n_output_tokens": agent_result.get("n_output_tokens"),
                "n_cache_tokens": agent_result.get("n_cache_tokens"),
                "cost_usd": agent_result.get("cost_usd"),
                "metadata": agent_result.get("metadata") or {},
            }
            stem = f"{_safe_name(candidate)}__{_safe_name(task_name)}__a{attempt}"
            raw_path = trace_dir / f"{stem}.raw_trajectory.json"
            evidence_path = trace_dir / f"{stem}.evidence.json"
            view_path = trace_dir / f"{stem}.md"
            if trajectory_source.exists():
                shutil.copyfile(trajectory_source, raw_path)
            else:
                raw_path.write_text("{}\n", encoding="utf-8")
            evidence = {
                "candidate": candidate,
                "task_name": task_name,
                "attempt": attempt,
                "status": status,
                **_trajectory_facts(trajectory),
            }
            evidence_path.write_text(
                json.dumps(evidence, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            view_path.write_text(
                _trace_markdown(
                    candidate,
                    task_name,
                    attempt,
                    trajectory,
                    status,
                    raw_path.name,
                    evidence_path.name,
                ),
                encoding="utf-8",
            )
            observations.append(
                TrialObservation(
                    task_name,
                    attempt,
                    _reward(result),
                    status,
                    raw_path,
                    view_path,
                    evidence_path,
                )
            )
    return observations


def observe_candidate(
    candidate: str,
    import_path: str,
    tasks: list[str],
    trace_dir: Path,
    *,
    dataset: str,
    model: str,
    environment: str,
    concurrency: int,
    max_turns: int,
    harbor_command: str,
    timeout: int,
    workdir: Path,
) -> list[TrialObservation]:
    """Run one candidate once per task and publish only sanitized agent traces."""
    staging_root = workdir / ".tthe_tmp"
    staging_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="tthe_tb2_harbor_", dir=staging_root
    ) as temp:
        jobs_dir = Path(temp) / "jobs"
        job_name = f"observe-{_safe_name(candidate)}"
        cmd = build_harbor_command(
            harbor_command,
            import_path,
            job_name,
            jobs_dir,
            tasks,
            dataset=dataset,
            model=model,
            environment=environment,
            concurrency=min(max(1, concurrency), len(tasks)),
            max_turns=max_turns,
        )
        env = direct_env(model)
        existing_pythonpath = env.get("PYTHONPATH")
        env["PYTHONPATH"] = str(workdir) + (
            os.pathsep + existing_pythonpath if existing_pythonpath else ""
        )
        process = subprocess.Popen(
            cmd,
            cwd=str(workdir),
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        try:
            stdout, stderr = process.communicate(timeout=timeout)
            completed = subprocess.CompletedProcess(
                cmd, process.returncode, stdout, stderr
            )
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                stdout, stderr = process.communicate(timeout=60)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                stdout, stderr = process.communicate()
            except ProcessLookupError:
                stdout, stderr = process.communicate()
            completed = subprocess.CompletedProcess(
                cmd, 124, stdout, (stderr or "") + "\nHarbor wall timeout"
            )
        except KeyboardInterrupt:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.communicate()
            except ProcessLookupError:
                process.communicate()
            raise
        job_dir = jobs_dir / job_name
        observations = collect_harbor_observations(job_dir, candidate, tasks, trace_dir)
        if all(
            observation.status.get("exception_type") == "MissingTrial"
            for observation in observations
        ):
            detail = (
                completed.stderr or completed.stdout or "Harbor produced no trial"
            )[-2000:]
            raise RuntimeError(
                f"Harbor failed before creating a trial (exit={completed.returncode}): "
                f"{detail}"
            )
        setup_failures = (
            "Failed to start tmux session",
            "failed to send non-blocking keys",
        )
        if observations and all(
            any(
                marker in str(observation.status.get("exception_message") or "")
                for marker in setup_failures
            )
            for observation in observations
        ):
            raise RuntimeError("Harbor agent setup failed before producing a trace")
        if observations and all(
            not (observation.status.get("n_input_tokens") or 0)
            and len(_read_json(observation.raw_trace_path).get("steps") or []) <= 1
            for observation in observations
        ):
            raise RuntimeError("Harbor produced a zero-token setup trace")
        if completed.returncode not in (0, 124) and all(
            o.reward is None for o in observations
        ):
            detail = (completed.stderr or completed.stdout or "Harbor failed")[-1000:]
            for obs in observations:
                obs.status["controller_error"] = detail
        return observations
