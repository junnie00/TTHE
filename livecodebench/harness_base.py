"""The HARNESS interface for LiveCodeBench (analogue of text_to_sql/harness_base.py).

A harness is ARBITRARY PYTHON wrapping the FROZEN weak solver. `solve()` can do anything the proposer
writes: read the problem, call the coder, RUN candidate code on the PUBLIC sample tests, diagnose
failures, repair, retry, self-consistency-vote, etc. LABEL-FREE: it may only look at the problem text and
PUBLIC-test execution results — never the private/hidden tests (those are gold, used outside for scoring).
"""
from abc import ABC, abstractmethod

from . import lcb_bridge as bridge


class CodeHarness(ABC):
    """Subclass this. You may rewrite ANY part of a harness — control flow, prompts, thinking, output cap,
    number of solver calls, self-verification, voting, repair loops — and `import bridge` to override any
    call-layer parameter. Your action space is the whole Python file, NOT just the helpers below. Two
    invariants are fixed (audit_harness.py checks them; a candidate that breaks either is invalid):

      * FROZEN SOLVER — never change WHO answers: no new client, no reassigning bridge._client /
        _SOLVER_MODEL, no base_url / api_key, no importing a network library. You MAY freely change HOW
        you call it (prompt, thinking, max_tokens, number of calls, voting).
      * LABEL-FREE — the line is the ANSWER KEY, not the inputs. You MAY run code on any inputs — public
        samples, self-generated stress inputs, and the real hidden test INPUTS via self.run_hidden(code)
        (this is transductive test-time adaptation) — and use crash / timeout / output-shape / cross-candidate
        agreement as signal. You must NEVER use the expected outputs / correctness: no bridge.is_correct, no
        reading a hidden test's `output` field, no per-question hardcoding of answers. Running on an input and
        seeing HOW it executes is label-free; comparing the output to the gold answer is not.

    Convenience helpers (use or ignore):
        self.problem        — the Problem (qid, content, starter_code, platform, difficulty, public_tests)
        self.content        — problem statement text
        self.starter_code   — '' for stdin problems; a signature for functional ones
        self.public_tests   — list of {input, output, testtype} (a LABEL-FREE signal)
        self.llm(prompt, system='', thinking=False, n=1, max_tokens=None)  — the frozen solver. thinking is a
            real trade-off with no preset best (decide from the traces): it often helps correctness on hard
            problems, but on a slow endpoint it can time out or emit no code. max_tokens=None is the default
            cap; raise it if replies get cut off before the fence.
        self.run_public(code) -> {n_pass, n_total, results}      — run code on the PUBLIC sample tests
        self.stress(code) -> {n_robust, n_total, results}        — run code on self-generated MAX-constraint inputs
        self.run_hidden(code) -> {n_ran, n_crash, n_timeout, n_empty, results}  — run code on the REAL hidden
            test INPUTS; EXECUTION status only (never the expected output / correctness). Strongest label-free
            robustness signal; `results[i].stdout` is the code's own output (usable for self-consistency).
        bridge.extract_code(text) -> str                         — pull a ```python``` block from a reply
    solve() must return the final program string.
    """

    def __init__(self, problem):
        self.problem = problem
        self.content = problem.content
        self.starter_code = problem.starter_code
        self.public_tests = problem.public_tests
        self._trace = []          # FULL trace: every coder call (+ thinking choice) + every public-test run

    def llm(self, prompt, system="", thinking=False, n=1, max_tokens=None):
        """Call the frozen coder. thinking=False -> no thinking (fast, but weaker on hard problems);
        'low'/'medium'/'high' -> think at that effort (often better on hard, slower / can time out on a slow
        endpoint). No preset best — decide from the traces. max_tokens=None -> the default output cap; raise it
        if replies get cut off before the code fence. Both choices are recorded in the trace for the proposer."""
        out = bridge.solver_llm(prompt, system=system, n=n, thinking=thinking, max_tokens=max_tokens)
        self._trace.append({"step": "coder_llm", "thinking": thinking, "max_tokens": max_tokens,
                            "system": system, "prompt": prompt,
                            "response": out if isinstance(out, str) else list(out)})
        return out

    def run_public(self, code):
        """Run candidate code on the PUBLIC sample tests (label-free). Records results into the trace."""
        res = bridge.run_code(code, self.public_tests)
        self._trace.append({"step": "run_public", "n_pass": res["n_pass"], "n_total": res["n_total"],
                            "results": res["results"]})
        return res

    def run_hidden(self, code):
        """Run `code` on the HIDDEN test INPUTS and get EXECUTION status only — {n_ran, n_crash, n_timeout,
        n_empty, results:[{status, stdout, ...}]}. This is transductive test-time adaptation on the *unlabeled*
        test inputs: it NEVER sees the expected outputs or whether the answer is correct (that stays in
        measurement). A crash/timeout here means the code is wrong on the hidden suite — a strictly stronger
        signal than stress() (real hidden inputs, not self-generated). `stdout` is the CODE's own output (not
        the answer), usable for self-consistency (do independent candidates agree?). Records into the trace."""
        res = bridge.run_hidden_inputs(code, self.problem)
        self._trace.append({"step": "run_hidden", "n_total": res["n_total"], "n_ran": res["n_ran"],
                            "n_crash": res["n_crash"], "n_timeout": res["n_timeout"], "n_empty": res["n_empty"]})
        return res

    def stress(self, code):
        """Run `code` on auto-generated MAX-constraint / boundary inputs (no expected output): catches the
        large-N TLE, integer-overflow, and float-precision bugs that the few small public tests miss. On a
        TIMEOUT/CRASH here, the solution is very likely wrong on the hidden suite -- switch to a faster or
        numerically-correct algorithm. Records into the trace."""
        res = bridge.run_stress(code, bridge.gen_stress_inputs(self.problem))
        nrob = sum(r["status"] == "ok" for r in res)
        self._trace.append({"step": "stress", "n_robust": nrob, "n_total": len(res),
                            "statuses": [r["status"] for r in res]})
        return {"n_robust": nrob, "n_total": len(res), "results": res}

    @abstractmethod
    def solve(self) -> str:
        """Return a complete, self-contained Python 3 program (reads stdin, writes stdout)."""
        ...
