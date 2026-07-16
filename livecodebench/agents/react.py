"""Plain ReAct LiveCodeBench harness — the execution-retry baseline.

The first model call uses bare's exact prompt, but with THINKING ENABLED and a
generous token budget, so the baseline solver is given its full reasoning
capacity (a fair reference, not a thinking-off handicap). The generated program
is then run on the public sample inputs; only a CRASH (non-zero exit —
exception or timeout) triggers another attempt, with the crash output fed back.
A program that runs to completion is submitted as-is: the sample *outputs* are
never checked for correctness, and there is no proactive probing or stress
testing. Reacting to wrong-but-running outputs is a behaviour the harness
evolution is meant to discover, not part of the baseline.
"""
from ..harness_base import CodeHarness
from .. import lcb_bridge as bridge


MAX_ATTEMPTS = 3
MAX_FEEDBACK_CHARS = 600

SYS = ("You are an expert competitive programmer. Read the problem and output ONE complete, self-contained "
       "Python 3 program that reads from standard input and prints the answer to standard output, inside a "
       "single ```python ... ``` block. No explanation outside the code block.")


class ReactHarness(CodeHarness):
    def solve(self) -> str:
        prompt = f"{self.content}\n\nWrite the complete Python 3 solution (read stdin, print stdout)."
        code = ""
        for _ in range(MAX_ATTEMPTS):
            resp = self.llm(prompt, system=SYS, thinking="high")
            code = bridge.extract_code(resp)
            if not code:
                prompt = (f"{self.content}\n\nYour previous reply contained no code block. "
                          "Output exactly one ```python ... ``` block with the complete solution.")
                continue
            res = self.run_public(code)
            crash = next((r for r in res.get("results", []) if r.get("rc") != 0), None)
            if res["n_total"] == 0 or crash is None:    # no tests, or ran to completion everywhere -> submit
                return code
            err = str(crash.get("stderr", "")).strip()[:MAX_FEEDBACK_CHARS] or "(non-zero exit, no stderr)"
            prompt = (f"{self.content}\n\nYour previous program crashed when run on a sample input:\n{err}\n\n"
                      "Fix the program and output the complete corrected Python 3 solution in one "
                      "```python ... ``` block.")
        return code
