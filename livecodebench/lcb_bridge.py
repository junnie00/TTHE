"""Bridge to LiveCodeBench (analogue of text_to_sql/bridge.py for the CODE domain).

Reuses the ase stack's FROZEN weak solver (deepseek-v4-flash). Lightweight — NO
Docker: execution = a subprocess running the candidate Python with the test `input` on stdin, comparing
stdout. Mirrors the SQL bridge so the same test-time-evolution orchestration can drive it.

Label-free signal = PUBLIC sample tests (shipped in each problem).  Gold (MEASUREMENT ONLY) = PRIVATE
hidden tests.  Data: livecodebench/code_generation_lite test<N>.jsonl pulled straight from the HF hub
(the modern `datasets` lib dropped script-dataset support, so we read the jsonl directly).
"""
import os
import re
import sys
import json
import base64
import zlib
import pickle
import subprocess
from dataclasses import dataclass

# Monorepo layout: this file is <REPO_ROOT>/livecodebench/lcb_bridge.py, so REPO_ROOT is two levels up.
# The shared `ase` package lives at <REPO_ROOT>/ase and is importable when running from the repo root.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
import time                                                                   # noqa: E402
import yaml                                                                   # noqa: E402
from openai import OpenAI                                                      # noqa: E402
from ase.llm import extract_code                                              # noqa: E402
from huggingface_hub import hf_hub_download                                  # noqa: E402

_CONFIG_PATH = os.environ.get("TTHE_CONFIG", os.path.join(REPO_ROOT, "config.yaml"))
_cfg = yaml.safe_load(open(_CONFIG_PATH, encoding="utf-8"))["llm"]
# Dedicated solver client: same FROZEN weak model (deepseek-v4-flash) + native THINKING, but with
# a MAX_TOKENS cap + timeout. HARD problems make this weak model think to the token limit and emit NO answer
# (finish=length, ~0 code) — the cap bounds latency so a runaway 'think forever' call can't stall the loop.
_client = OpenAI(base_url=_cfg["base_url"], api_key=os.environ[_cfg["api_key_env"]],
                 timeout=float(os.environ.get("LCB_SOLVE_TIMEOUT", "600")), max_retries=0)
_SOLVER_MODEL = _cfg["solver_model"]
_REASONING = os.environ.get("LCB_REASONING_EFFORT", _cfg.get("reasoning_effort", "high"))
_MAX_TOKENS = int(os.environ.get("LCB_MAX_TOKENS", "32000"))   # generous so thinking-on never truncates before the code fence
# HOW this endpoint expresses thinking (config, not model-name guessing): "deepseek" -> send the
# {"thinking": {...}} extra_body (deepseek-v*, mimo-v*); "none" -> no toggle, use temperature.
_THINKING_STYLE = _cfg.get("thinking_style", "deepseek")
_TEMP = float(_cfg.get("temperature", 0.0))


def solver_llm(prompt, system="", n=1, thinking=False, max_tokens=None):
    """FROZEN weak solver. THINKING IS HARNESS-CONTROLLED: thinking=False -> disabled (sane default — the
    weak model over-thinks on hard and emits NO code); 'low'|'medium'|'high' -> enabled at that effort.
    max_tokens is HARNESS-CONTROLLED too (None -> LCB_MAX_TOKENS default); the output cap bounds a
    think-forever call. n=1 -> str, n>1 -> list[str]."""
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    mt = _MAX_TOKENS if max_tokens is None else int(max_tokens)
    kw = dict(model=_SOLVER_MODEL, messages=msgs, max_tokens=mt)
    if _THINKING_STYLE == "deepseek":                # deepseek-v* / mimo-v* native thinking toggle
        if thinking:
            eff = thinking if isinstance(thinking, str) else _REASONING
            kw["extra_body"] = {"thinking": {"type": "enabled"}, "reasoning_effort": eff}
        else:
            kw["extra_body"] = {"thinking": {"type": "disabled"}}
    else:                                            # no thinking toggle -> temperature only
        kw["temperature"] = _TEMP

    def one():
        for attempt in range(2):
            try:
                r = _client.chat.completions.create(**kw)
                msg = r.choices[0].message
                # A gateway that ignores thinking:disabled can route the whole reply into
                # reasoning_content, leaving content empty — fall back to it instead of dropping the work.
                return msg.content or getattr(msg, "reasoning_content", "") or ""
            except Exception:  # noqa: BLE001
                if attempt == 1:
                    return ""
                time.sleep(2.0)
    outs = [one() for _ in range(max(n, 1))]
    return outs[0] if n == 1 else outs


extract_code = extract_code        # re-export ase's ```python``` extractor


def back_translate(code):
    """Plain-English description of what `code` ACTUALLY computes -- an intent-level (label-free) signal
    surfaced in the trace so the proposer/judge can compare it to the problem. Frozen solver, thinking off."""
    if not code:
        return "(no code produced)"
    prompt = ("Here is a Python solution program:\n```python\n" + code[:3000] + "\n```\n\n"
              "In 1-3 sentences, describe in plain English what this program reads from standard input and "
              "what it computes and prints. Describe what the code ACTUALLY does, not what it may be intended "
              "to do.")
    try:
        return solver_llm(prompt, system="You concisely explain what a program does.", thinking=False).strip()[:600]
    except Exception:  # noqa: BLE001
        return "(back-translation unavailable)"


# ---------- constraint-aware stress testing (robustness signal, no expected outputs) ----------
_STRESS_CACHE = {}
_STRESS_SYS = ("You write tiny self-contained Python 3 snippets that each PRINT exactly one valid input "
               "(in the required format) for a competitive-programming problem.")


_STRESS_MAX_CHARS = 8_000_000    # 2e5 numbers @ ~11 chars each ~= 2.2MB; generous headroom above that


def gen_stress_inputs(problem, k=3, gen_timeout=15):
    """The model reads the problem's CONSTRAINTS and writes k input-GENERATORS producing EXTREME / boundary
    inputs, which we execute to materialize input strings. Label-free: inputs only, no expected outputs.
    These catch the large-N TLE / overflow / precision bugs that the few small public tests hide. Cached/qid.

    An oversized generator output is DROPPED, never truncated: a truncated input (header says N=300000 but
    the data is cut mid-stream) is INVALID and makes even a gold-correct solution crash — that one bug
    poisoned the whole stress signal with false 'not robust' verdicts."""
    if problem.qid in _STRESS_CACHE:
        return _STRESS_CACHE[problem.qid]
    if getattr(problem, "starter_code", ""):
        fmt = ("The input format for this FUNCTION-STYLE problem is: one JSON literal PER LINE, one line per "
               "function argument (e.g. a list arg prints as [1, 2, 3] on its own line, an int arg as 7).")
    else:
        fmt = "Print the input in the exact stdin format the problem statement specifies."
    prompt = (f"{problem.content[:3000]}\n\nWrite {k} separate tiny Python 3 snippets. Each snippet must "
              f"PRINT exactly ONE valid input for THIS problem to stdout. {fmt} "
              f"Make them STRESS a solution: (1) the MAXIMUM sizes/values allowed by the constraints; (2) a "
              f"minimal/boundary case; (3) another large adversarial case. Each snippet self-contained; print "
              f"nothing but the input; put each in its own ```python``` block.")
    try:
        resp = solver_llm(prompt, system=_STRESS_SYS, thinking=False)
    except Exception:  # noqa: BLE001
        resp = ""
    inputs = []
    for s in re.findall(r"```python\s*(.*?)```", resp, re.S)[:k]:
        rc, out, _ = _run_one(s, "", gen_timeout)
        if rc == 0 and out.strip() and len(out) <= _STRESS_MAX_CHARS:
            inputs.append(out)
    _STRESS_CACHE[problem.qid] = inputs
    return inputs


def run_stress(code, inputs, timeout=15):
    """Run CODE on each stress input (no expected output) -- a robustness probe for TLE / crash / empty
    output that the small public tests miss. Returns [{status, out, err}].

    timeout raised 5->15: on this box a CORRECT Python solution needs several seconds just for I/O+sort at
    N=2e5, so 5s flagged gold-correct code as TIMEOUT (false positive). A real TLE (e.g. O(N^2) at large N)
    still blows well past 15s."""
    res = []
    for inp in inputs:
        rc, so, se = _run_one(code, inp, timeout)
        st = "TIMEOUT" if rc == -9 else ("CRASH" if rc != 0 else ("EMPTY-OUTPUT" if not so.strip() else "ok"))
        res.append({"status": st, "out": so[:80], "err": se[:100]})
    return res


# ---------- dataset ----------
def _decode_tests(raw):
    """public_test_cases = plain JSON list; private_test_cases = base64 -> zlib -> pickle -> JSON string."""
    try:
        return json.loads(raw)
    except Exception:
        dec = pickle.loads(zlib.decompress(base64.b64decode(raw)))
        return json.loads(dec) if isinstance(dec, str) else dec


@dataclass
class Problem:
    qid: str
    title: str
    content: str
    starter_code: str
    platform: str
    difficulty: str
    contest_date: str
    public_tests: list
    _raw_private: str = ""
    _priv_cache: list = None

    def private_tests(self):
        if self._priv_cache is None:
            self._priv_cache = _decode_tests(self._raw_private) if self._raw_private else []
        return self._priv_cache

    @property
    def testtype(self):
        return self.public_tests[0]["testtype"] if self.public_tests else "stdin"


def load_problems(version="test6", difficulty=None, stdin_only=True, limit=None, recent_first=True):
    """Pull <version>.jsonl from the HF hub and build Problem objects (private tests decoded lazily)."""
    path = hf_hub_download("livecodebench/code_generation_lite", f"{version}.jsonl", repo_type="dataset")
    probs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            probs.append(Problem(d["question_id"], d["question_title"], d["question_content"],
                                 d.get("starter_code", ""), d["platform"], d["difficulty"],
                                 d.get("contest_date", ""), _decode_tests(d["public_test_cases"]),
                                 d["private_test_cases"]))
    if difficulty:
        probs = [p for p in probs if p.difficulty == difficulty]
    if stdin_only:
        probs = [p for p in probs if p.public_tests and p.public_tests[0]["testtype"] == "stdin"]
    if recent_first:
        probs.sort(key=lambda p: p.contest_date, reverse=True)
    return probs[:limit] if limit else probs


# ---------- execution (stdin-type) ----------
def _norm(s):
    return "\n".join(ln.rstrip() for ln in (s or "").strip().splitlines())


def _run_one(code, stdin, timeout):
    try:
        r = subprocess.run([sys.executable, "-c", code], input=stdin or "", capture_output=True,
                           text=True, timeout=timeout)
        return r.returncode, r.stdout, r.stderr
    except subprocess.TimeoutExpired:
        return -9, "", "TIMEOUT"
    except Exception as e:  # noqa: BLE001
        return -1, "", str(e)[:200]


# ---------- execution (functional / LeetCode-type) ----------
# A functional problem gives a `class Solution` signature instead of a stdin program: each test's `input`
# is one JSON literal PER LINE (the call arguments) and `output` is the JSON of the expected return value.
# We wrap the candidate in a driver that parses the args off stdin, calls the method, and prints the JSON
# result — so the same subprocess/timeout machinery as the stdin path is reused.
_METHOD_RE = re.compile(r"def\s+(\w+)\s*\(\s*self")


def method_name(starter_code):
    """The Solution method a functional problem must implement ('' if not a functional problem)."""
    m = _METHOD_RE.search(starter_code or "")
    return m.group(1) if m else ""


def _functional_driver(code, meth):
    return (
        "from typing import List, Optional, Dict, Set, Tuple, Any\n"
        "import json as _json, sys as _sys\n"
        f"{code}\n\n"
        "_args = [_json.loads(_l) for _l in _sys.stdin.read().split('\\n') if _l.strip() != '']\n"
        f"_out = Solution().{meth}(*_args)\n"
        "print(_json.dumps(_out))\n"
    )


def _json_eq(got, expected):
    try:
        return json.loads(got) == json.loads(expected)
    except Exception:  # noqa: BLE001
        return _norm(got) == _norm(expected)


def _run_functional(code, meth, test_input, timeout):
    return _run_one(_functional_driver(code, meth), test_input, timeout)


def _testtype(tests):
    return tests[0].get("testtype", "stdin") if tests else "stdin"


def run_code(code, tests, timeout=8, starter_code=""):
    """Run candidate CODE against `tests`. stdin problems: CODE is a self-contained stdin->stdout program.
    functional problems (starter_code gives a `class Solution` signature): CODE implements that class and we
    call the method with the test's JSON args, comparing the JSON return value. Returns
    {n_pass, n_total, results:[{ok, rc, stdout, stderr, input, expected}]}. Never raises."""
    functional = _testtype(tests) == "functional"
    meth = method_name(starter_code) if functional else ""
    results = []
    for t in tests:
        if functional and meth:
            rc, out, err = _run_functional(code, meth, t.get("input", ""), timeout)
            ok = (rc == 0) and _json_eq(out, t.get("output", ""))
        else:
            rc, out, err = _run_one(code, t.get("input", ""), timeout)
            ok = (rc == 0) and (_norm(out) == _norm(t.get("output", "")))
        results.append({"ok": ok, "rc": rc, "stdout": out[:400], "stderr": err[:300],
                        "input": str(t.get("input", ""))[:160], "expected": str(t.get("output", ""))[:160]})
    return {"n_pass": sum(r["ok"] for r in results), "n_total": len(results), "results": results}


def is_correct(code, problem, timeout=6):
    """MEASUREMENT ONLY: True iff CODE passes ALL private (hidden) tests. Short-circuits on the FIRST
    failing test so a wrong/slow solution doesn't run the whole (possibly large, slow) hidden suite.
    Handles both stdin and functional problems."""
    priv = problem.private_tests()
    if not priv or not code:
        return False
    functional = _testtype(priv) == "functional" or problem.testtype == "functional"
    meth = method_name(problem.starter_code) if functional else ""
    if functional and not meth:
        return False                                    # cannot call an unknown method — never score it True
    for t in priv:
        if functional:
            rc, out, _ = _run_functional(code, meth, t.get("input", ""), timeout)
            if rc != 0 or not _json_eq(out, t.get("output", "")):
                return False
        else:
            rc, out, _ = _run_one(code, t.get("input", ""), timeout)
            if rc != 0 or _norm(out) != _norm(t.get("output", "")):
                return False
    return True
