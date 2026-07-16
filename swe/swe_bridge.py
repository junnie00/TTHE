"""Bridge to SWE-bench Verified (analogue of livecodebench/lcb_bridge.py for the SWE domain).

The FROZEN solver = a FULL mini-swe-agent rollout (deepseek-v4-flash via litellm) looping bash commands in
the instance's Docker container until it submits a git-diff patch. Label-free signal = the agent's own
message TRAJECTORY (reproduction scripts + test runs) + the final PATCH. Gold scoring (MEASUREMENT ONLY) =
the official `swebench.harness.run_evaluation` harness -> resolved bool, cached in logs/gold_cache.json.
"""
import hashlib
import json
import os
import random
import subprocess
from pathlib import Path

from datasets import load_dataset

from minisweagent.agents.default import DefaultAgent
from minisweagent.models.litellm_model import LitellmModel
from minisweagent.config import builtin_config_dir, get_config_from_spec
from minisweagent.run.benchmarks.swebench import get_sb_environment

HERE = Path(__file__).parent
LOGS = HERE / "logs"
BASE_URL = os.environ.get("SWE_SOLVER_BASE_URL", "https://api.deepseek.com/v1")   # any OpenAI-compatible endpoint
MODEL = os.environ.get("SWE_SOLVER_MODEL", "openai/deepseek-chat")               # litellm model id (openai/<name>)
# HOW this endpoint expresses thinking (env, not model-name guessing): "deepseek" -> send the
# {"thinking": {...}} extra_body (deepseek-v*, mimo-v*); "none" -> no toggle (e.g. plain OpenAI).
THINKING_STYLE = os.environ.get("SWE_THINKING_STYLE", "deepseek")
DATASET = "princeton-nlp/SWE-Bench_Verified"

# Load the default swebench.yaml config ONCE; keep its stock prompts as the defaults.
_CFG = get_config_from_spec(str(builtin_config_dir / "benchmarks" / "swebench.yaml"))
DEFAULT_SYS = _CFG["agent"]["system_template"]
DEFAULT_INST = _CFG["agent"]["instance_template"]


def load_instances(ids=None, shuffle_seed=42, limit=None):
    """Load SWE-bench Verified. If `ids` given, return those instances IN THAT ORDER; else replicate the
    official slice: sort by instance_id, seed(42), shuffle, then [:limit]. Returns list of instance dicts."""
    rows = {x["instance_id"]: x for x in load_dataset(DATASET, split="test")}
    if ids is not None:
        return [rows[i] for i in ids]
    order = sorted(rows.keys())
    random.seed(shuffle_seed)
    random.shuffle(order)
    if limit:
        order = order[:limit]
    return [rows[i] for i in order]


def agent_rollout(instance, system_template=None, instance_template=None, step_limit=80, wall_time=1500):
    """ONE frozen mini-swe-agent rollout in the instance's Docker container (custom prompts optional ->
    fall back to the stock swebench.yaml ones). Returns (patch, info) where info has exit_status, n_calls,
    messages. Never raises; cleans up the container best-effort."""
    env = None
    try:
        env = get_sb_environment(_CFG, instance)
        model_kwargs = {"api_base": BASE_URL,
                        "api_key": os.environ.get("SWE_SOLVER_API_KEY") or os.environ.get("OPENAI_API_KEY", ""),
                        "drop_params": True}
        if THINKING_STYLE == "deepseek":   # deepseek-v* / mimo-v* native thinking toggle; else no toggle
            # explicit THINKING-ON, consistent with the other domains (reasoning_effort is largely ignored
            # by some endpoints, but we set it so the config is unambiguous).
            model_kwargs["extra_body"] = {"thinking": {"type": "enabled"}, "reasoning_effort": "high"}
        model = LitellmModel(model_name=MODEL, model_kwargs=model_kwargs, cost_tracking="ignore_errors")
        agent = DefaultAgent(
            model, env,
            system_template=system_template or DEFAULT_SYS,
            instance_template=instance_template or DEFAULT_INST,
            step_limit=step_limit, cost_limit=0.0, wall_time_limit_seconds=wall_time,
        )
        result = agent.run(instance["problem_statement"])
        patch = result.get("submission", "") or ""
        return patch, {"exit_status": result.get("exit_status"), "n_calls": agent.n_calls,
                       "messages": agent.messages}
    except Exception as e:  # noqa: BLE001
        return "", {"exit_status": type(e).__name__, "n_calls": 0, "messages": []}
    finally:
        for m in ("cleanup", "__del__"):
            try:
                getattr(env, m, lambda: None)()
            except Exception:  # noqa: BLE001
                pass


def is_correct_batch(items, run_id):
    """MEASUREMENT ONLY. `items` = list of (instance, patch). Run the official swebench harness on the
    uncached ones; return {instance_id: resolved_bool}. Cached in logs/gold_cache.json by patch hash."""
    LOGS.mkdir(parents=True, exist_ok=True)
    cache_path = LOGS / "gold_cache.json"
    cache = json.load(open(cache_path)) if cache_path.exists() else {}
    out, todo = {}, []
    for inst, patch in items:
        iid = inst["instance_id"]
        if not patch:
            out[iid] = False
            continue
        key = f"{iid}:{hashlib.sha1(patch.encode()).hexdigest()[:12]}"
        if key in cache:
            out[iid] = cache[key]
        else:
            todo.append((inst, patch, key))

    if todo:
        preds = [{"instance_id": inst["instance_id"], "model_name_or_path": "tt", "model_patch": patch}
                 for inst, patch, _ in todo]
        preds_path = LOGS / f"preds_{run_id}.json"
        preds_path.write_text(json.dumps(preds, indent=2))
        iids = [inst["instance_id"] for inst, _, _ in todo]
        subprocess.run(
            ["python", "-m", "swebench.harness.run_evaluation", "--dataset_name", DATASET,
             "--predictions_path", str(preds_path), "--run_id", run_id, "--instance_ids", *iids,
             "--max_workers", str(min(len(iids), 8)), "--cache_level", "instance"],
            cwd=str(LOGS), check=False)
        # Prefer the per-instance report.json; fall back to the summary tt.{run_id}.json (resolved_ids).
        summary = LOGS / f"tt.{run_id}.json"
        resolved_ids = set()
        if summary.exists():
            resolved_ids = set(json.load(open(summary)).get("resolved_ids", []))
        for inst, patch, key in todo:
            iid = inst["instance_id"]
            report = LOGS / "run_evaluation" / run_id / "tt" / iid / "report.json"
            if report.exists():
                ok = bool(json.load(open(report)).get(iid, {}).get("resolved", False))
            else:
                ok = iid in resolved_ids
            cache[key] = ok
            out[iid] = ok
        json.dump(cache, open(cache_path, "w"), indent=2)
    return out


def is_correct(instance, patch, run_id):
    """MEASUREMENT ONLY: single-item wrapper over is_correct_batch."""
    return is_correct_batch([(instance, patch)], run_id).get(instance["instance_id"], False)
