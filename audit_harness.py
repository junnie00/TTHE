"""Audit an evolved harness against the TTHE invariants.

TTHE evolves the harness inside a bounded action space: the proposer may rewrite anything about *how* the
solver is called and how the answer is built, but must not touch the two invariants that define the method:

  1. FROZEN SOLVER  — the model / weights / endpoint / key are fixed. A harness may change the prompt,
     thinking effort, max_tokens, number of calls, voting, verification, etc., but must NOT construct a new
     client, swap the model name, or point at a different endpoint/key.
  2. LABEL-FREE     — the harness may read only label-free signals (public tests, execution, self-check,
     back-translation). It must NOT read gold / hidden tests / the grading answer.

This is a static (AST + token) checker. It is deliberately conservative: it flags anything that *looks* like
a violation so a human can confirm. Use it as a post-hoc audit over evolved candidates, or wire `audit_file`
into an optimizer to reject a candidate before it is ever committed.

    python -m audit_harness path/to/agents/            # scan a dir of harnesses
    python -m audit_harness path/to/cand_x.py          # scan one file
Exit code is 1 if any violation is found.
"""
import argparse
import ast
import sys
from pathlib import Path

# --- FROZEN SOLVER: names that mean "I am changing WHO answers", not "how I call it" ---
FROZEN_ATTRS = {"_client", "_SOLVER_MODEL", "_ctrl_client"}         # mutating these swaps the solver
FROZEN_CALLS = {"OpenAI", "AsyncOpenAI", "Anthropic", "AsyncAnthropic"}  # newing up a client
FROZEN_KWARGS = {"base_url", "api_key", "api_base"}                 # redirecting the endpoint/key
FROZEN_IMPORTS = {"openai", "anthropic", "httpx", "requests", "urllib", "aiohttp", "socket"}

# --- LABEL-FREE: names that mean "I am reading the answer key" ---
GOLD_NAMES = {"private_tests", "gold", "gold_sql", "gold_result", "is_correct", "code_context",
              "reference_code", "hidden_tests", "expected_output", "grade", "rubric_answer"}


def audit_source(src, filename="<harness>"):
    """Return a list of {rule, line, detail} violations for one harness source string."""
    out = []
    try:
        tree = ast.parse(src, filename=filename)
    except SyntaxError as e:
        return [{"rule": "PARSE", "line": e.lineno or 0, "detail": f"cannot parse: {e.msg}"}]

    for node in ast.walk(tree):
        # import openai / requests / ... (a harness has no business calling the network itself)
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] in FROZEN_IMPORTS:
                    out.append({"rule": "FROZEN_SOLVER", "line": node.lineno,
                                "detail": f"imports network/client lib '{a.name}'"})
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] in FROZEN_IMPORTS:
                out.append({"rule": "FROZEN_SOLVER", "line": node.lineno,
                            "detail": f"imports from '{node.module}'"})
        # assigning to bridge._client / _SOLVER_MODEL (swap the solver)
        elif isinstance(node, ast.Assign):
            for tgt in node.targets:
                for sub in ast.walk(tgt):
                    if isinstance(sub, ast.Attribute) and sub.attr in FROZEN_ATTRS:
                        out.append({"rule": "FROZEN_SOLVER", "line": node.lineno,
                                    "detail": f"assigns to solver internal '.{sub.attr}'"})
                    if isinstance(sub, ast.Name) and sub.id in FROZEN_ATTRS:
                        out.append({"rule": "FROZEN_SOLVER", "line": node.lineno,
                                    "detail": f"rebinds solver internal '{sub.id}'"})
        # OpenAI(...) / redirected endpoint kwargs
        elif isinstance(node, ast.Call):
            fn = node.func
            fname = fn.attr if isinstance(fn, ast.Attribute) else (fn.id if isinstance(fn, ast.Name) else "")
            if fname in FROZEN_CALLS:
                out.append({"rule": "FROZEN_SOLVER", "line": node.lineno,
                            "detail": f"constructs a client '{fname}(...)'"})
            for kw in node.keywords:
                if kw.arg in FROZEN_KWARGS:
                    out.append({"rule": "FROZEN_SOLVER", "line": node.lineno,
                                "detail": f"passes endpoint kwarg '{kw.arg}='"})
        # reading gold / hidden tests / grading answer — ONLY via attribute access on self/bridge
        # (self.problem.private_tests(), bridge.is_correct(...)). A bare local named `is_correct` or
        # `code_context` is the candidate's own variable, not a read of the answer key.
        if isinstance(node, ast.Attribute) and node.attr in GOLD_NAMES:
            base = node.value
            base_name = (base.id if isinstance(base, ast.Name)
                         else base.attr if isinstance(base, ast.Attribute) else "")
            if base_name in ("self", "bridge", "problem") or base_name.endswith("bridge"):
                out.append({"rule": "LABEL_FREE", "line": node.lineno,
                            "detail": f"reads gold signal '{base_name}.{node.attr}'"})

    # dedupe (same rule+line+detail can be hit twice by nested walks)
    seen, uniq = set(), []
    for v in out:
        k = (v["rule"], v["line"], v["detail"])
        if k not in seen:
            seen.add(k); uniq.append(v)
    return sorted(uniq, key=lambda v: v["line"])


def audit_file(path):
    return audit_source(Path(path).read_text(encoding="utf-8", errors="ignore"), str(path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="harness .py files or dirs of them")
    ap.add_argument("--quiet", action="store_true", help="only print files WITH violations")
    args = ap.parse_args()

    files = []
    for p in args.paths:
        p = Path(p)
        files += sorted(p.rglob("*.py")) if p.is_dir() else [p]
    files = [f for f in files if f.name != "__init__.py"]

    n_bad = n_parse = 0
    for f in files:
        v = audit_file(f)
        real = [x for x in v if x["rule"] != "PARSE"]
        parse = [x for x in v if x["rule"] == "PARSE"]
        if real:
            n_bad += 1
            print(f"✗ {f}")
            for x in real:
                print(f"    [{x['rule']}] line {x['line']}: {x['detail']}")
        elif parse:
            n_parse += 1
            if not args.quiet:
                print(f"~ {f}  (syntax error — not an invariant violation)")
        elif not args.quiet:
            print(f"✓ {f}")
    print(f"\n{len(files)} scanned, {n_bad} invariant violation(s), {n_parse} unparseable.")
    sys.exit(1 if n_bad else 0)


if __name__ == "__main__":
    main()
