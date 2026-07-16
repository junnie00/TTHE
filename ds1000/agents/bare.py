"""Baseline harness: one greedy shot (THINKING OFF), problem -> the solution snippet. No self-check, no
retry, no adaptation. This is the floor every proposed harness is measured against. (The proposer may turn
thinking on, add self-test-driven repair, etc.)"""
from ..harness_base import DS1000Harness
from .. import ds1000_bridge as bridge

SYS = ("You are an expert Python data-science programmer. Read the problem and output ONLY the solution code "
       "that should be INSERTED to compute the required `result` variable, USING the input variables already "
       "defined in the problem's context (e.g. df, a, X). Do NOT redefine or re-create those input variables, "
       "do NOT add your own example/test data, do NOT wrap the answer in a function — just the lines that compute "
       "`result` from the given inputs. Put it in a single ```python ... ``` block, no prose.")


class BareHarness(DS1000Harness):
    def solve(self) -> str:
        resp = self.llm(self.prompt, system=SYS)        # thinking=False (default)
        return bridge.extract_code(resp)
