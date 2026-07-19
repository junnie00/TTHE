"""The HARNESS interface for DS-1000 (analogue of livecodebench/harness_base.py).

A harness wraps a FROZEN data-science coder (deepseek-flash) that writes pandas/numpy/scipy/sklearn/torch/tf
snippets. `solve()` returns the solution code: the snippet that, given the problem's context variables in
scope, sets `result`. LABEL-FREE: it may use ONLY the problem prompt and the SELF-CHECK execution (construct
an example input from the prompt, run the candidate, observe whether it runs and what it produces) and
back-translation — NEVER the gold `code_context` / hidden test (those are used outside, for scoring only).

The proposer evolves: prompt engineering, self-test-driven repair (construct input from the prompt example,
run, fix on error/mismatch), thinking control, and robust code extraction.
"""
from abc import ABC, abstractmethod

from . import ds1000_bridge as bridge


class DS1000Harness(ABC):
    """Subclass this. You may rewrite ANY part of a harness — control flow, prompts, thinking, output cap,
    number of solver calls, self-verification, voting, repair loops — and `import bridge` to override any
    call-layer parameter. Your action space is the whole Python file, NOT just the helpers below. Two
    invariants are fixed (audit_harness.py checks them; a candidate that breaks either is invalid):

      * FROZEN SOLVER — never change WHO answers: no new client, no reassigning bridge._client /
        _SOLVER_MODEL, no base_url / api_key, no importing a network library. You MAY freely change HOW
        you call it (prompt, thinking, max_tokens, number of calls, voting).
      * LABEL-FREE — read only label-free signals (the prompt, self-check execution). Never read gold:
        self.problem's code_context / the hidden test, bridge.is_correct, or any grading answer.

    Convenience helpers (use or ignore):
        self.prompt         — the NL problem statement (often with a worked input->output example)
        self.library        — pandas / numpy / scipy / sklearn / pytorch / tensorflow / matplotlib
        self.llm(prompt, system='', thinking=False, n=1, max_tokens=None)  — the frozen solver. thinking=False
            (default) is fast; 'low'|'medium'|'high' enables it. max_tokens=None is the default cap.
        self.run(script, timeout=15) -> (rc, stdout, stderr)  — LABEL-FREE arbitrary execution. Build your
            OWN evidence with it: several constructed inputs, differential tests between two solutions,
            self-written invariant assertions. Never put the gold test in it.
        self.selfcheck(code) -> {checkable, ran, error, output}  — LABEL-FREE probe. checkable=False means it
            could not be BUILT (no concrete example input in the prompt) = NO EVIDENCE, not a failure
        bridge.extract_code(text) -> str                   — pull a ```python``` block from a reply
    solve() must return the final solution-snippet string.
    """

    def __init__(self, problem):
        self.problem = problem
        self.prompt = problem.prompt
        self.library = problem.library
        self._trace = []          # FULL trace: every coder call (+ thinking choice) + every self-check
        self._call_seq = {}       # (request signature) -> how many times THIS instance already issued it

    def llm(self, prompt, system="", thinking=False, n=1, max_tokens=None):
        """Call the frozen coder. thinking=False -> no thinking (fast); 'low'/'medium'/'high' -> think at that
        effort. max_tokens=None -> the default output cap; raise it if replies get cut off before the code
        fence. Both choices are recorded in the trace so the proposer can see/evolve them."""
        # seq = how many times THIS harness instance already issued this exact request. Asking the same thing
        # twice on purpose (e.g. to vote) still gets two different replies; asking it once gets the same reply
        # any other harness would get, so identical prompts can no longer produce spurious score differences.
        sig = (prompt, system, str(thinking), max_tokens, n)
        seq = self._call_seq.get(sig, 0)
        self._call_seq[sig] = seq + 1
        out = bridge.solver_llm(prompt, system=system, n=n, thinking=thinking, max_tokens=max_tokens, seq=seq)
        self._trace.append({"step": "coder_llm", "thinking": thinking, "max_tokens": max_tokens,
                            "system": system, "prompt": prompt,
                            "response": out if isinstance(out, str) else list(out)})
        return out

    def run(self, script, timeout=15):
        """LABEL-FREE general execution: run any Python `script` you compose -> (rc, stdout, stderr).

        `selfcheck` answers exactly one fixed question. This answers whatever question you can write code for,
        and it is how you MANUFACTURE EVIDENCE instead of only consuming it. Examples of signals it makes
        reachable, none of which need an answer key:
          * ROBUSTNESS — run one solution on several inputs you construct from the problem prose; a solution
            that works on one shape and crashes or degenerates on another is suspect.
          * DIFFERENTIAL — ask the coder twice (or ask for two different approaches), run BOTH on the same
            input and diff the results. Disagreement proves at least one is wrong. Agreement is weak evidence
            (the coder's errors correlate), never proof.
          * SELF-WRITTEN ASSERTIONS — encode invariants you can read off the problem text (output length,
            dtype, monotonicity, that a filter never grows the input) and execute them.
        Every call is recorded in the trace. NEVER put the gold test in here — no `code_context`, no
        `is_correct`, no `reference_code`; audit_harness.py rejects a candidate that does."""
        rc, out, err = bridge.run_script(script, timeout)
        self._trace.append({"step": "run", "script": script, "rc": rc, "stdout": out, "stderr": err})
        return rc, out, err

    def selfcheck(self, code):
        """LABEL-FREE: run the candidate against the problem's own <code> input setup and observe whether it
        runs, what `result` it produces, and whether it HARDCODES (redefines) the input variables. Records
        into the trace. Never uses gold."""
        sc = bridge.selfcheck(code, self.problem)
        self._trace.append({"step": "selfcheck", "checkable": sc.get("checkable", True),
                            "ran": sc["ran"], "error": sc["error"], "output": sc["output"],
                            "redefines": sc.get("redefines", [])})
        return sc

    @abstractmethod
    def solve(self) -> str:
        """Return the solution code: the snippet that, given the problem's context variables, sets `result`."""
        ...
