"""Bridge between the TTHE loop and the SQL execution/data/LLM stack (the `ase` package).

Provides BIRD loading (ase.dataset), SQLite execution + result comparison (ase.db), and the frozen
solver LLM over any OpenAI-compatible endpoint (ase.llm). Importing this module reads the run config
(config.yaml at the repo root, or the path in $TTHE_CONFIG) and builds one shared LLM + dataset.

Config keys (see config.example.yaml): `llm` (base_url / api_key_env / solver_model / controller_model),
`dataset` (name: demo|bird, bird_root, db_id), and `output_dir`. The solver endpoint/model can also be
overridden per-process via SOLVER_BASE_URL / SOLVER_MODEL / SOLVER_API_KEY_ENV for model-ablation runs.
"""
import os
import sys

import yaml

_THIS = os.path.abspath(__file__)
REPO_ROOT = os.path.dirname(os.path.dirname(_THIS))                           # repo root (parent of tthe/)
if REPO_ROOT not in sys.path:                                                 # so the sibling `ase` package resolves
    sys.path.insert(0, REPO_ROOT)
CONFIG_PATH = os.environ.get("TTHE_CONFIG", os.path.join(REPO_ROOT, "config.yaml"))

from ase.solver_cache import SolverCache
from ase.llm import LLM, LLMConfig, extract_sql          # noqa: E402
from ase.dataset import build_dataset                    # noqa: E402
from ase.db import compare_results                       # noqa: E402

EXEC_TIMEOUT, EXEC_LIMIT = 30.0, 20000

_cfg = yaml.safe_load(open(CONFIG_PATH, encoding="utf-8"))
# --- solver model-ablation overrides (env only; config.yaml left untouched). When SOLVER_MODEL is set we
# point BOTH the solver and controller roles at that one model, so a model-ablation run is pure (no
# gpt-5.1 / gateway dependency). Unset -> config.yaml is used as-is. ---
_llmcfg = dict(_cfg["llm"])
if os.environ.get("SOLVER_BASE_URL"):
    _llmcfg["base_url"] = os.environ["SOLVER_BASE_URL"]
if os.environ.get("SOLVER_MODEL"):
    _llmcfg["solver_model"] = _llmcfg["controller_model"] = os.environ["SOLVER_MODEL"]
if os.environ.get("SOLVER_API_KEY_ENV"):
    _llmcfg["api_key_env"] = _llmcfg["controller_api_key_env"] = os.environ["SOLVER_API_KEY_ENV"]
_LLM = LLM(LLMConfig(**_llmcfg))
_out = _cfg.get("output_dir", "runs")
_out = _out if os.path.isabs(_out) else os.path.join(REPO_ROOT, _out)
_DS = build_dataset(_cfg["dataset"], _out)

extract_sql = extract_sql        # re-export
compare_results = compare_results


def get_db(db_id):
    return _DS.get_database(db_id)


def eval_questions(db_id):
    return _DS.eval_questions(db_id)


import threading as _threading
_tls = _threading.local()


def set_temp_override(t):
    """Thread-local temperature override for solver_llm (None clears). Lets us run a harness at T>0 for
    SELF-CONSISTENCY without touching the harness's own temperature=0 calls."""
    _tls.temp_override = t


_CACHE = SolverCache(os.environ.get("SQL_SOLVER_CACHE",
                                    os.path.join(os.path.dirname(__file__), "logs", "solver_cache.json")))


def solver_llm(prompt, system="", temperature=0.0, n=1, seq=0):
    """Call the FROZEN weak solver (deepseek-v4-flash). n=1 -> str, n>1 -> list[str].

    Replies are CACHED on (prompt, system, temperature, n, seq) — see ase.solver_cache. NOTE: caching is
    keyed on the EFFECTIVE temperature, so a self-consistency harness running at T>0 via set_temp_override
    still draws distinct samples through `seq`, exactly as intended."""
    ov = getattr(_tls, "temp_override", None)
    t = ov if ov is not None else temperature
    # Some model APIs constrain temperature to a single value (for example,
    # kimi-k2.6 accepts only 1). Keep the default harness behavior unchanged
    # unless an isolated model-ablation process explicitly opts in.
    if os.environ.get("SOLVER_TEMPERATURE_OVERRIDE"):
        t = float(os.environ["SOLVER_TEMPERATURE_OVERRIDE"])
    outs = _CACHE.get_or_call((prompt, system, t, n, seq),
                              lambda: _LLM.chat("harness", system, prompt, model_role="solver", n=n,
                                                temperature=t))
    return outs[0] if n == 1 else outs


def execute(db, sql):
    """Run SQL; returns {ok, rows, ...}. Never raises."""
    return db.execute(sql, timeout=EXEC_TIMEOUT, limit=EXEC_LIMIT)


def gold_result(db, gold_sql):
    return db.execute(gold_sql, timeout=EXEC_TIMEOUT, limit=EXEC_LIMIT)


def is_correct(pred_result, gold):
    """MEASUREMENT ONLY. True iff the predicted rows match the gold rows (set semantics)."""
    return bool(pred_result["ok"] and gold["ok"] and compare_results(pred_result["rows"], gold["rows"]))
