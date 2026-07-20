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

import sys as _sys
_sys.path.insert(0, str(Path(__file__).parent.parent))
from ase.solver_cache import SolverCache

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


# ── FROZEN-SOLVER primitives (the ONLY thing frozen is the LLM: model / endpoint / weights). The harness
# owns everything else — the bash-interaction LOOP, the tool use, prompts, budget, post-rollout verify. ──

def make_env(instance):
    """Start the instance's real repo Docker container. Caller owns teardown via teardown_env()."""
    return get_sb_environment(_CFG, instance)


def teardown_env(env):
    for m in ("cleanup", "__del__"):
        try:
            getattr(env, m, lambda: None)()
        except Exception:  # noqa: BLE001
            pass


_CACHE = SolverCache(os.environ.get("SWE_SOLVER_CACHE", str(LOGS / "solver_cache.json")))


class _CachedModel:
    """LitellmModel wrapper that caches every model reply, keyed on the FULL message list.

    Why it works here, and why it is worth more here than anywhere else: an agent turn is determined
    entirely by the conversation so far, so two harnesses that send the same messages get the same reply
    and therefore run the same command and see the same output — the ENTIRE rollout replays from cache.
    A TTHE batch runs many candidates that share a base and often identical templates; without this each
    one pays full price for a trajectory it shares with its siblings, and SWE is by far the most expensive
    domain (median 63 model calls per instance, up to ~150).

    It also removes the same measurement noise the other domains got from ase.solver_cache: two harnesses
    that behave identically now SCORE identically, so a score difference means a real behavioural
    difference. (Measured on DS-1000 before caching: byte-identical prompts scored 6/10, 3/10 and 4/10.)

    Divergence degrades gracefully — the moment a command output differs, every later prompt differs and
    those calls simply miss."""

    def __init__(self, inner):
        self._inner = inner
        self._seq = {}

    def __getattr__(self, name):          # cost, n_calls, config, ... stay on the wrapped model
        return getattr(self._inner, name)

    def query(self, messages, **kw):
        key = json.dumps(messages, ensure_ascii=False, sort_keys=True, default=str)
        # seq keeps a harness that deliberately re-asks the SAME question from collapsing into one answer.
        n = self._seq.get(key, 0)
        self._seq[key] = n + 1
        return _CACHE.get_or_call((MODEL, key, n), lambda: self._inner.query(messages, **kw))


def make_model():
    """The FROZEN weak solver (deepseek-v4-flash / mimo-v* via litellm). Do NOT vary model/endpoint."""
    model_kwargs = {"api_base": BASE_URL,
                    "api_key": os.environ.get("SWE_SOLVER_API_KEY") or os.environ.get("OPENAI_API_KEY", ""),
                    "drop_params": True}
    if THINKING_STYLE == "deepseek":                     # deepseek-v* / mimo-v* native thinking toggle
        model_kwargs["extra_body"] = {"thinking": {"type": "enabled"}, "reasoning_effort": "high"}
    return _CachedModel(LitellmModel(model_name=MODEL, model_kwargs=model_kwargs,
                                     cost_tracking="ignore_errors"))


def model_query(model, messages):
    """ONE call to the frozen solver. `messages` = [{role, content}, ...]. Returns the assistant text."""
    try:
        return (model.query(messages) or {}).get("content", "")
    except Exception:  # noqa: BLE001
        return ""


def exec_in_repo(env, command, timeout=None):
    """Run one bash `command` in the instance's REAL repo container. Returns {output, returncode}.
    LABEL-FREE: this is the actual repo — apply a patch, run the agent's OWN reproduction script, run the
    repo's EXISTING test suite (regression). Never run the gold FAIL_TO_PASS/PASS_TO_PASS suite as a
    pass/fail verdict — that is the answer key (is_correct, measurement-only)."""
    try:
        r = env.execute({"command": command}, timeout=timeout)
        return {"output": r.get("output", ""), "returncode": r.get("returncode", -1)}
    except Exception as e:  # noqa: BLE001
        return {"output": f"(exec error: {type(e).__name__}: {str(e)[:200]})", "returncode": -1}


STEP_LIMIT = 80        # FIXED — see run_stock_agent.__doc__; harnesses may not change it


def run_stock_agent(env, model, instance, system_template=None, instance_template=None,
                    wall_time=5400):
    """Run ONE stock mini-swe-agent rollout on a GIVEN env+model (caller owns their lifecycle). This is the
    default bash loop — a CONVENIENCE the harness may keep, replace with its own llm+exec loop, or wrap.

    STEP_LIMIT IS FIXED AT 80 AND IS NOT A HARNESS PARAMETER. mini-swe-agent's swebench.yaml specifies
    250, and 250 was measured here: of 17 instances, 12 finished within 80 steps (identical either way)
    and 5 exceeded it. Those 5 cost roughly 10x more — an agent turn resends the whole conversation, so
    spend grows with the SQUARE of the step count — and bought exactly ONE extra solve (2/17 at 250 vs
    1/17 at 80). The worst offender spent 306 steps and still returned an empty patch: it was looping,
    not converging.

    It is fixed rather than merely defaulted because it is otherwise a lever the PROPOSER could pull. A
    harness that simply raised the budget would post a gain that is bought with money, not with a better
    method, and the comparison against the baseline would be meaningless. Budget is held constant so that
    what evolves is the strategy. (A harness may still spend its budget better — noticing a looping
    rollout and stopping early is a real and reachable improvement.)

    wall_time has no official counterpart; it exists only so a wedged rollout cannot stall the loop. It was
    1500s, which at the restored 250-step budget would simply have become the new binding limit (a smoke
    run spent ~10 minutes reaching step 80), moving the handicap rather than removing it. 5400s is loose
    enough to bind only on a genuinely stuck agent."""
    try:
        agent = DefaultAgent(
            model, env,
            system_template=system_template or DEFAULT_SYS,
            instance_template=instance_template or DEFAULT_INST,
            step_limit=STEP_LIMIT, cost_limit=0.0, wall_time_limit_seconds=wall_time,
        )
        result = agent.run(instance["problem_statement"])
        return (result.get("submission", "") or ""), {"exit_status": result.get("exit_status"), "error": "",
                                                       "n_calls": agent.n_calls, "messages": agent.messages}
    except Exception as e:  # noqa: BLE001
        # Keep the MESSAGE, not just the class. Everything that goes wrong outside the agent's control —
        # an exhausted API budget, an unreachable endpoint, a dead container — arrives here, and with only
        # the class name an infrastructure failure is indistinguishable from "the model could not do it":
        # both surface as an empty patch. Measured: a run where the endpoint's rolling budget ran out
        # produced 8 instances with an empty patch and zero steps, which without this text reads as eight
        # model failures. `RateLimitError: ... ExceededBudget: Key over 3h budget` says what it really was.
        return "", {"exit_status": type(e).__name__, "error": str(e)[:1500],
                    "n_calls": 0, "messages": []}


def agent_rollout(instance, system_template=None, instance_template=None, wall_time=5400):
    """Backward-compatible one-shot: make env+model, run the stock agent, tear the container down."""
    env = None
    try:
        env = make_env(instance)
        return run_stock_agent(env, make_model(), instance, system_template, instance_template,
                               wall_time)
    except Exception as e:  # noqa: BLE001
        return "", {"exit_status": type(e).__name__, "n_calls": 0, "messages": []}
    finally:
        if env is not None:
            teardown_env(env)


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
