"""Bridge to DS-1000 (analogue of livecodebench/lcb_bridge.py for the DATA-SCIENCE coding domain).

Reuses the ase stack's FROZEN weak solver (deepseek-v4-flash). Lightweight — NO Docker:
execution = a subprocess running the candidate snippet inside the problem's context. Mirrors the LCB bridge
so the same test-time-evolution orchestration can drive it.

Label-free signal = a SELF-CHECK driver the frozen model writes from the PROMPT only (construct a plausible
example input, run the candidate, observe whether it runs and what `result` it produces). Gold (MEASUREMENT
ONLY) = the shipped `code_context` test harness. Data: load_dataset("xlangai/DS-1000", split="test").
"""
import ast
import os
import re
import sys
import subprocess
import tempfile
from dataclasses import dataclass

# Monorepo root = parent of this domain's directory. The shared `ase` package lives at REPO_ROOT/ase.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
import time                                                                   # noqa: E402
import yaml                                                                   # noqa: E402
from openai import OpenAI                                                      # noqa: E402
from ase.llm import extract_code as _raw_extract_code                         # noqa: E402


def extract_code(text):
    """ase extractor + strip stray language-tag / fence lines (`python`, ```` ``` ````) the model sometimes
    leaves inside the block, which would otherwise corrupt the snippet (SyntaxError / NameError)."""
    code = _raw_extract_code(text) or ""
    keep = [ln for ln in code.splitlines() if ln.strip() not in ("```", "```python", "python", "py")]
    return "\n".join(keep).strip()


_CONFIG_PATH = os.environ.get("TTHE_CONFIG", os.path.join(REPO_ROOT, "config.yaml"))
_cfg = yaml.safe_load(open(_CONFIG_PATH, encoding="utf-8"))["llm"]
# Dedicated solver client: same FROZEN weak model (deepseek-v4-flash) + native THINKING, but with
# a MAX_TOKENS cap + timeout. Built LAZILY so this module imports with only a dummy key present.
_client = None
_SOLVER_MODEL = _cfg["solver_model"]
_REASONING = os.environ.get("DS1000_REASONING_EFFORT", _cfg.get("reasoning_effort", "high"))
_MAX_TOKENS = int(os.environ.get("DS1000_MAX_TOKENS", "65536"))   # 32000 was measured TOO SMALL on LCB
# (there, a hard problem converged only at 43k reasoning tokens and returned NOTHING under a 32000 cap — a
#  budget starvation we had mis-read as "the model cannot solve it". Same solver here, so same cap.)
# HOW this endpoint expresses thinking (config, not model-name guessing): "deepseek" -> send the
# {"thinking": {...}} extra_body (deepseek-v*, mimo-v*); "none" -> no toggle, use temperature.
_THINKING_STYLE = _cfg.get("thinking_style", "deepseek")
_TEMP = float(_cfg.get("temperature", 0.0))


def _get_client():
    global _client
    if _client is None:
        _client = OpenAI(base_url=_cfg["base_url"], api_key=os.environ[_cfg["api_key_env"]],
                         timeout=float(os.environ.get("DS1000_SOLVE_TIMEOUT", "1200")), max_retries=0)
    return _client


def solver_llm(prompt, system="", n=1, thinking=False, max_tokens=None):
    """FROZEN weak solver. THINKING IS HARNESS-CONTROLLED: thinking=False -> disabled (sane default — the
    weak model over-thinks and emits NO code); 'low'|'medium'|'high' -> enabled at that effort.
    max_tokens is HARNESS-CONTROLLED too (None -> DS1000_MAX_TOKENS default); the output cap bounds a
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
                r = _get_client().chat.completions.create(**kw)
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
    prompt = ("Here is a Python data-science snippet that computes a `result` variable from some surrounding "
              "context variables:\n```python\n" + code[:3000] + "\n```\n\n"
              "In 1-3 sentences, describe in plain English what this snippet computes and assigns to `result`. "
              "Describe what the code ACTUALLY does, not what it may be intended to do.")
    try:
        return solver_llm(prompt, system="You concisely explain what a snippet does.", thinking=False).strip()[:600]
    except Exception:  # noqa: BLE001
        return "(back-translation unavailable)"


# ---------- dataset ----------
@dataclass
class Problem:
    pid: str
    prompt: str
    code_context: str
    library: str
    reference_code: str = ""        # GOLD — NEVER shown to harness/proposer; metadata only
    difficulty: str = "hard"        # settable by the loop; unused for scoring


def load_problems(ids=None, min_ref_lines=None, limit=None):
    """Load DS-1000 test; build a Problem per row keyed by metadata['problem_id'].
    `ids` -> return exactly those problem_ids; elif `min_ref_lines` -> filter by reference_code line count."""
    from datasets import load_dataset
    ds = load_dataset("xlangai/DS-1000", split="test")
    by_pid = {}
    for row in ds:
        md = row["metadata"]
        md = md if isinstance(md, dict) else eval(md)
        pid = str(md["problem_id"])
        by_pid[pid] = Problem(pid=pid, prompt=row["prompt"], code_context=row["code_context"],
                              library=md["library"], reference_code=row["reference_code"])
    if ids is not None:
        ids = [str(i) for i in ids]
        return [by_pid[i] for i in ids if i in by_pid]
    probs = list(by_pid.values())
    if min_ref_lines:
        probs = [p for p in probs if len(p.reference_code.splitlines()) >= min_ref_lines]
    return probs[:limit] if limit else probs


# ---------- execution ----------
def _run_script(script, timeout):
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(script)
        path = f.name
    try:
        r = subprocess.run([sys.executable, path], capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout, r.stderr
    except subprocess.TimeoutExpired:
        return -9, "", "TIMEOUT"
    except Exception as e:  # noqa: BLE001
        return -1, "", str(e)[:200]
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def is_correct(solution, problem, timeout=20):
    """MEASUREMENT ONLY: run the GOLD `code_context` test harness against the candidate `solution` in a
    subprocess. test_execution(solution) raises AssertionError if wrong, returns None if correct. Never raises."""
    if not solution:
        return False
    script = problem.code_context + "\n\n__sol = " + repr(solution) + "\ntest_execution(__sol)\nprint('PASS')\n"
    rc, out, _ = _run_script(script, timeout)
    return "PASS" in out


# ---------- LABEL-FREE self-check ----------
_SELFCHECK_SYS = ("You write tiny self-contained Python 3 drivers. Given a data-science problem and a candidate "
                  "solution snippet (which sets a `result` variable from context variables described in the "
                  "problem), you emit ONE driver that constructs a PLAUSIBLE example input matching the problem "
                  "(reuse any worked example shown in the problem), defines those context variables, runs the "
                  "candidate snippet, and prints `result`. Output ONLY the driver in a single ```python``` block.")


def _redefines_input(setup, sol):
    """Names of the problem's INPUT variables (defined in the <code> setup) that `sol` overwrites with a
    fresh value NOT derived from the variable itself — i.e. the solution HARDCODES its own example data
    instead of using the provided input. Such code runs in the self-check (the hardcoded values match the
    example) but FAILS the hidden test (which supplies different inputs). `df = df.sort_values()` is fine
    (RHS references df); `a = np.array([...])` is a redefinition (RHS does not reference a)."""
    try:
        setup_vars = {t.id for n in ast.walk(ast.parse(setup)) if isinstance(n, ast.Assign)
                      for t in n.targets if isinstance(t, ast.Name)}
        bad = []
        for n in ast.walk(ast.parse(sol)):
            if isinstance(n, ast.Assign):
                for t in n.targets:
                    if isinstance(t, ast.Name) and t.id in setup_vars:
                        refs = {x.id for x in ast.walk(n.value) if isinstance(x, ast.Name)}
                        if t.id not in refs:
                            bad.append(t.id)
        return sorted(set(bad))
    except SyntaxError:
        return []


def _prompt_setup(problem):
    """The example input-setup the model is shown. Normally a <code>...</code> block; some problems omit the
    closing tag, so fall back to everything before the solution marker."""
    m = re.search(r"<code>(.*?)</code>", problem.prompt, re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r"<code>(.*)", problem.prompt, re.DOTALL)
    if not m:
        return ""
    body = m.group(1)
    for marker in ("### BEGIN SOLUTION", "BEGIN SOLUTION", "# SOLUTION START"):
        if marker in body:
            body = body.split(marker)[0]
            break
    return _strip_trailing_stub(body.strip())


def _strip_trailing_stub(setup):
    """Drop a trailing, bodyless `def f(df=example_df):` stub from an extracted setup.

    Function-body problems end their prompt with the stub the solution is meant to fill. Keeping it makes the
    setup end inside an open block, so anything appended afterwards is parsed as that function's body and
    raises IndentationError. Only the concrete input definitions above it are wanted here — the real function
    header comes from the benchmark's own exec_context template."""
    lines = setup.splitlines()
    for i, ln in enumerate(lines):
        if re.match(r"^\s*def\s+\w+\s*\(", ln):
            rest = [x for x in lines[i + 1:] if x.strip() and not x.strip().startswith("#")]
            if not rest:                       # nothing but comments after the header -> it is a stub
                return "\n".join(lines[:i]).rstrip()
    return setup


def _exec_template(problem):
    """The benchmark's own CALLING CONVENTION for this problem: the `exec_context` string, which shows where
    the solution is inserted (e.g. inside `def f(df):`) and how `result` is produced. This is STRUCTURE only —
    it holds no expected answer (those live in generate_ans / define_test_input, which we never touch)."""
    m = re.search(r'exec_context\s*=\s*r?"""(.*?)"""', problem.code_context, re.DOTALL)
    return m.group(1) if m else ""


def selfcheck(solution, problem, timeout=12):
    """LABEL-FREE health probe — DETERMINISTIC, no model-written driver.

    Runs the candidate THE WAY THE BENCHMARK WILL: the solution is placed at `[insert]` inside the problem's
    own `exec_context` template, so a function-body problem is executed as a function body, and `result` is
    produced by the template's own final line. The inputs come from the EXAMPLE setup shown in the prompt —
    never from define_test_input — and nothing is ever compared against generate_ans. Gold stays untouched.

    Why this matters (measured): running the snippet standalone instead let a solution that hardcodes its own
    copy of the input and never returns anything execute cleanly and be reported healthy, while inserting the
    SAME code into the real template produced None. That single mismatch is a large share of the ~61% of
    self-check-approved solutions that fail the hidden test. Falls back to the standalone form when the
    problem ships no usable template.

    Returns {ran, error, output, redefines, result_none_by_design}; `redefines` lists input variables the
    solution reassigns, and `result_none_by_design` is True for plotting problems whose template ends in
    `result = None` (graded from the saved figure), so an empty `result` there is expected, not a failure."""
    if not solution:
        return {"ran": False, "error": "(no solution code)", "output": "", "redefines": [],
                "result_none_by_design": False}
    setup = _prompt_setup(problem)
    sol = "\n".join(ln for ln in solution.splitlines() if ln.strip() not in ("```", "```python", "python"))
    redef = _redefines_input(setup, sol) if setup else []

    tmpl = _exec_template(problem)
    script = ""
    if tmpl and "[insert]" in tmpl:
        # Every name the template takes from the hidden test_input must be obtainable from the prompt's
        # example setup instead — either defined there directly, or as example_<name>. If any name cannot be
        # bound label-free, fall back rather than guess.
        ok = True
        for names in re.findall(r"^(.+?)\s*=\s*test_input\s*$", tmpl, re.M):
            for nm in [n.strip() for n in names.split(",")]:
                # A setup line like `df = load_data()` is a PLACEHOLDER: DS-1000 uses it to say "the data
                # arrives here" without shipping load_data, so treating it as a real definition leaves the
                # name unbound and the run dies with NameError. Only concrete assignments count.
                m_def = re.search(rf"^\s*{re.escape(nm)}\s*=\s*(.+)$", setup, re.M)
                concrete = bool(m_def) and "load_data(" not in m_def.group(1)
                if not (concrete or re.search(rf"^\s*example_{re.escape(nm)}\s*=", setup, re.M)):
                    ok = False
        if ok:
            # Substitute the hidden-input line IN PLACE (never delete it, never hoist the binding to the
            # top): its position matters. In a function-body problem the template reads
            #   def f(df):\n[insert]\ndf = test_input\nresult = f(df)
            # so `df = test_input` sits AFTER the function at module level — moving that assignment above
            # the def, or dropping the line and prepending the binding, breaks the indentation.
            def _bind_line(mo):
                names = [n.strip() for n in mo.group(1).split(",")]
                pairs = [f"{n} = example_{n}" for n in names
                         if not re.search(rf"^\s*{re.escape(n)}\s*=", setup, re.M)]
                return "\n".join(pairs)                      # '' when the setup already defines them
            body = re.sub(r"^(.+?)\s*=\s*test_input\s*$", _bind_line, tmpl, flags=re.M)
            body = body.replace("[insert]", sol)
            script = (f"{setup}\n{body}\n"
                      "try:\n    print(repr(result))\n"
                      "except NameError:\n    print('(no `result` variable set)')\n")
    if not script:                                              # no usable template -> previous behaviour
        script = f"{setup}\n{sol}\ntry:\n    print(repr(result))\nexcept NameError:\n    print('(no `result` variable set)')\n"

    # Plotting problems end their OWN template with `result = None` and are graded from the saved figure, so
    # a None result there is the template speaking, not a failing solution. Flag it, or every correct
    # Matplotlib answer reads as "produced nothing".
    none_by_design = bool(re.search(r"^\s*result\s*=\s*None\s*$", tmpl, re.M)) if tmpl else False

    rc, out, err = _run_script(script, timeout)
    if rc == 0:
        return {"ran": True, "error": "", "output": out[:1000], "redefines": redef,
                "result_none_by_design": none_by_design}
    return {"ran": False, "error": (("TIMEOUT" if rc == -9 else err) or "")[:600], "output": out[:1000],
            "redefines": redef, "result_none_by_design": none_by_design}
