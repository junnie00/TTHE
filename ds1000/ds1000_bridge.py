"""Bridge to DS-1000 (analogue of livecodebench/lcb_bridge.py for the DATA-SCIENCE coding domain).

Reuses the ase stack's FROZEN weak solver (deepseek-v4-flash). Lightweight — NO Docker:
execution = a subprocess running the candidate snippet inside the problem's context. Mirrors the LCB bridge
so the same test-time-evolution orchestration can drive it.

Label-free signal = a SELF-CHECK driver the frozen model writes from the PROMPT only (construct a plausible
example input, run the candidate, observe whether it runs and what `result` it produces). Gold (MEASUREMENT
ONLY) = the shipped `code_context` test harness. Data: load_dataset("xlangai/DS-1000", split="test").
"""
import ast
import threading
import hashlib
import json
import os
import re
import sys
import subprocess
import tempfile
import textwrap
from dataclasses import dataclass

# Monorepo root = parent of this domain's directory. The shared `ase` package lives at REPO_ROOT/ase.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
import time                                                                   # noqa: E402
import yaml                                                                   # noqa: E402
from openai import OpenAI                                                      # noqa: E402
from ase.llm import extract_code as _raw_extract_code                         # noqa: E402
from ase.solver_cache import SolverCache                                       # noqa: E402


def extract_code(text):
    """ase extractor + strip stray language-tag / fence lines (`python`, ```` ``` ````) the model sometimes
    leaves inside the block, which would otherwise corrupt the snippet (SyntaxError / NameError).

    REPAIR THE FIRST-LINE DEDENT. `ase.llm.extract_code` ends in `.strip()`, which strips the whole block
    string and therefore eats the leading whitespace of the FIRST LINE ONLY. A reply whose body is uniformly
    indented — very common on DS-1000's function-body problems, where the coder is asked for an indented
    block — arrives here with line 0 at column 0 and every other line still indented, which is a guaranteed
    `IndentationError: unexpected indent`. Correct solutions were destroyed by this and the harness then fed
    the syntax error back as though the model's algorithm were at fault.

    Restore line 0's indentation to match its successors, then dedent uniformly."""
    code = _raw_extract_code(text) or ""
    keep = [ln for ln in code.splitlines() if ln.strip() not in ("```", "```python", "python", "py")]
    if not keep:
        return ""
    rest = [ln for ln in keep[1:] if ln.strip()]
    if rest and not keep[0][:1].isspace() and keep[0].strip():
        pad = min(len(ln) - len(ln.lstrip()) for ln in rest)
        if pad > 0:                       # line 0 lost exactly this much to the upstream .strip()
            keep[0] = " " * pad + keep[0]
    return textwrap.dedent("\n".join(keep).strip("\n")).rstrip()


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


_CACHE = SolverCache(os.environ.get("DS1000_SOLVER_CACHE",
                                     os.path.join(os.path.dirname(__file__), "logs", "solver_cache.json")))


def solver_llm(prompt, system="", n=1, thinking=False, max_tokens=None, seq=0):
    """FROZEN weak solver. THINKING IS HARNESS-CONTROLLED: thinking=False -> disabled (sane default — the
    weak model over-thinks and emits NO code); 'low'|'medium'|'high' -> enabled at that effort.
    max_tokens is HARNESS-CONTROLLED too (None -> DS1000_MAX_TOKENS default); the output cap bounds a
    think-forever call. n=1 -> str, n>1 -> list[str].

    Replies are CACHED on (prompt, system, thinking, max_tokens, seq) — see _cache()."""
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
    outs = []
    for i in range(max(n, 1)):
        # Each of the n samples is its own cache slot: n>1 is a harness ASKING for diversity, so slot i must
        # stay distinct, while a second harness issuing the same n>1 request reuses the same i answers.
        outs.append(_CACHE.get_or_call((prompt, system, thinking, mt, seq, i), one))
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
_TMP_PATH_RE = re.compile(r'/tmp/tmp[A-Za-z0-9_]+\.py')


def _run_script(script, timeout):
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(script)
        path = f.name
    try:
        r = subprocess.run([sys.executable, path], capture_output=True, text=True, timeout=timeout)
        # Scrub the RANDOM temp-file name out of the traceback. A harness feeds this text back to the
        # solver as a retry prompt, so leaving it in made every retry prompt unique and therefore
        # permanently uncacheable — measured: 23 of 114 coder calls in one batch (20%) could never hit
        # the cache, precisely on the problems that need repair, i.e. where harnesses differ most.
        # The path also tells the model nothing it can use.
        return r.returncode, r.stdout, _TMP_PATH_RE.sub("<script>", r.stderr)
    except subprocess.TimeoutExpired:
        return -9, "", "TIMEOUT"
    except Exception as e:  # noqa: BLE001
        return -1, "", _TMP_PATH_RE.sub("<script>", str(e))[:200]
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def run_script(script, timeout=15):
    """LABEL-FREE general execution: run an arbitrary Python `script` in a subprocess -> (rc, stdout, stderr).

    This is the primitive a harness needs to MANUFACTURE ITS OWN EVIDENCE, and its absence is why every
    evolved candidate so far only tinkered with prompts and retry logic: `selfcheck` was the sole execution
    the API offered, `ran` was therefore the only observable, and the only failure mode that both moves `ran`
    and is fixable label-free is a syntax/indentation error. Everything deeper — decomposition, planning,
    self-written tests — produced no observable at all, so it could never be selected for.

    With this, a harness can build signals of its own: run one solution on several self-constructed inputs and
    check it does not crash or degenerate; generate TWO independent solutions and DIFF their outputs on the
    same input (disagreement proves at least one is wrong, with no gold involved); write and execute its own
    assertions. It carries no answer key — what is executed is whatever the harness composed."""
    return _run_script(script, timeout)


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


def _example_assignments(setup):
    """name -> the COMPLETE source of the statement assigning it, in the prompt's example setup.

    Must be AST-based, not line-based: DS-1000 example inputs are routinely multi-line
    (`df = pd.DataFrame({\\n  'a': [...],\\n})`), and grabbing only the first line yields a
    syntactically broken fragment — measured to DOUBLE the false-kill rate versus doing nothing."""
    try:
        tree = ast.parse(setup)
    except SyntaxError:
        return {}
    out = {}
    for order, node in enumerate(tree.body):
        if isinstance(node, ast.Assign):
            src = ast.get_source_segment(setup, node)
            if not src:
                continue
            deps = {n.id for n in ast.walk(node.value) if isinstance(n, ast.Name)}
            for t in node.targets:
                if isinstance(t, ast.Name):
                    out[t.id] = {"src": src, "deps": deps, "order": order}
    return out


def _example_closure(examples, names):
    """Sources needed to define `names`, INCLUDING their transitive dependencies, in setup order.

    Example setups chain: `d = {...}` then `df = pd.DataFrame(d)`. Emitting only the `df` line leaves `d`
    unbound (measured: 3 more gold solutions rejected with a NameError the solution had nothing to do with).
    Returns (sources, unresolved-names)."""
    need, seen, missing = set(), set(), []
    stack = list(names)
    while stack:
        nm = stack.pop()
        if nm in seen:
            continue
        seen.add(nm)
        ent = examples.get(nm)
        if not ent:
            continue
        need.add(nm)
        for d in ent["deps"]:
            if d in examples and d not in seen:
                stack.append(d)
    srcs = [examples[n]["src"] for n in sorted(need, key=lambda n: examples[n]["order"])]
    return srcs, missing


# A setup line like `df = load_data()` is DS-1000 saying "the data arrives here"; the function is never
# shipped. Such a name has NO concrete example value, so the solution cannot be exercised label-free.
_PLACEHOLDER_CALL = re.compile(r"\b(load_data|load_iris|load_digits|fetch_\w+)\s*\(")



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

    THREE outcomes, not two. `checkable=False` means the probe could not be BUILT — the template's inputs
    have no concrete example value in the prompt (`df = load_data()` is DS-1000's placeholder for "the data
    arrives here"; the function is never shipped). That is ABSENCE OF EVIDENCE and must never be reported as
    `ran=False`: a harness that feeds a fabricated error back to the model makes it "fix" correct code.
    Measured on hard50: 16/50 problems are unbuildable this way, and the old code reported every one of them
    as an execution failure.

    Returns {checkable, ran, error, output, redefines, result_none_by_design}; `redefines` lists input
    variables the solution reassigns, and `result_none_by_design` is True for plotting problems whose template
    ends in `result = None` (graded from the saved figure), so an empty `result` there is expected."""
    if not solution:
        return {"checkable": True, "ran": False, "error": "(no solution code)", "output": "",
                "redefines": [], "result_none_by_design": False}
    setup = _prompt_setup(problem)
    sol = "\n".join(ln for ln in solution.splitlines() if ln.strip() not in ("```", "```python", "python"))
    redef = _redefines_input(setup, sol) if setup else []
    tmpl = _exec_template(problem)
    none_by_design = bool(re.search(r"^\s*result\s*=\s*None\s*$", tmpl, re.M)) if tmpl else False
    unbuildable = {"checkable": False, "ran": False, "error": "", "output": "", "redefines": redef,
                   "result_none_by_design": none_by_design}
    if not (tmpl and "[insert]" in tmpl):
        return {**unbuildable, "error": "(problem ships no exec_context template)"}

    # Build the probe from the TEMPLATE ALONE. The prompt's setup is NOT prepended: it restates the
    # template's own imports and intermediate variables, so concatenating the two executed everything twice
    # and tore open block structure (measured: 11/50 gold solutions rejected with IndentationError/NameError).
    # The template is already complete except for the hidden inputs — so bind ONLY those, from the example.
    examples = _example_assignments(setup)
    missing = []

    def _bind_line(mo):
        """Rewrite `df, y = test_input` in place, into concrete example assignments. Position matters: in a
        function-body problem the line sits AFTER the `def` at module level, so it must not be hoisted."""
        out = []
        for name in [n.strip() for n in mo.group(1).split(",")]:
            key = f"example_{name}" if f"example_{name}" in examples else name
            ent = examples.get(key)
            if not ent or _PLACEHOLDER_CALL.search(ent["src"]):
                missing.append(name)
                out.append(f"{name} = None")
                continue
            # Emit the whole dependency chain, then the binding itself under the template's name.
            srcs, _ = _example_closure(examples, [key])
            for s in srcs:
                out.append(s if s != ent["src"] else
                           re.sub(rf"^\s*(example_)?{re.escape(name)}\s*=", f"{name} =", s, count=1))
        return "\n".join(out)

    body = re.sub(r"^(.+?)\s*=\s*test_input\s*$", _bind_line, tmpl, flags=re.M)
    if missing:
        return {**unbuildable,
                "error": f"(no example value in the prompt for {sorted(set(missing))} — cannot probe label-free)"}
    script = (body.replace("[insert]", sol)
              + "\ntry:\n    print(repr(result))\nexcept NameError:\n    print('(no `result` variable set)')\n")

    rc, out, err = _run_script(script, timeout)
    if rc != 0:
        # A NameError for a template-supplied name means the probe itself is malformed, not the solution.
        if "NameError" in err and not re.search(r"NameError.*'(" + "|".join(map(re.escape, examples)) + r")'", err):
            return {**unbuildable, "error": f"(probe could not be built: {err.strip().splitlines()[-1][:200]})"}
        return {"checkable": True, "ran": False, "error": (("TIMEOUT" if rc == -9 else err) or "")[:600],
                "output": out[:1000], "redefines": redef, "result_none_by_design": none_by_design}
    # Producing no `result` (or a None one, outside plotting problems) is a REAL negative: the template ran
    # to completion and the solution still set nothing. Previously this returned ran=True.
    empty = "(no `result` variable set)" in out or out.strip() == "None"
    if empty and not none_by_design:
        return {"checkable": True, "ran": False, "error": "(solution set no `result`)", "output": out[:1000],
                "redefines": redef, "result_none_by_design": False}
    return {"checkable": True, "ran": True, "error": "", "output": out[:1000], "redefines": redef,
            "result_none_by_design": none_by_design}
