"""Plain ReAct DS-1000 harness — the execution-retry baseline.

The first model call uses bare's exact prompt, but with THINKING ENABLED and a
generous token budget, so the baseline solver is given its full reasoning
capacity (a fair reference, not a thinking-off handicap). The generated snippet
is then executed on a constructed example context; only an execution ERROR
triggers another attempt, with the error fed back. A snippet that runs is
submitted as-is: there is no checking of output correctness, no
input-redefinition heuristic, and no proactive probing — those are behaviours
the harness evolution is meant to discover, not part of the baseline.
"""
from ..harness_base import DS1000Harness
from .. import ds1000_bridge as bridge


MAX_ATTEMPTS = 3

SYS = ("You are an expert Python data-science programmer. Read the problem and output ONLY the solution code "
       "that should be INSERTED to compute the required `result` variable, USING the input variables already "
       "defined in the problem's context (e.g. df, a, X). Do NOT redefine or re-create those input variables, "
       "do NOT add your own example/test data, do NOT wrap the answer in a function — just the lines that compute "
       "`result` from the given inputs. Put it in a single ```python ... ``` block, no prose.")


class ReactHarness(DS1000Harness):
    def solve(self) -> str:
        prompt = self.prompt
        code = ""
        for _ in range(MAX_ATTEMPTS):
            resp = self.llm(prompt, system=SYS, thinking="high")
            code = bridge.extract_code(resp)
            if not code:
                prompt = (self.prompt + "\n\nYour previous reply contained no code block. Output exactly one "
                          "```python ... ``` block computing `result`.")
                continue
            sc = self.selfcheck(code)
            # `checkable=False` = the probe could not be BUILT (the prompt gives no concrete example input),
            # so there is NO evidence either way. Retrying on it would feed the model a fabricated error and
            # make it "fix" code that was never shown to be broken — submit as-is instead.
            if sc["ran"] or not sc.get("checkable", True):
                return code
            prompt = (self.prompt + "\n\nYour previous snippet raised an error when executed on the example "
                      "input:\n" + str(sc.get("error", ""))[:600]
                      + "\n\nFix it and output ONLY the corrected snippet (just the lines that compute "
                      "`result`) in one ```python ... ``` block.")
        return code
