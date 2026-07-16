"""Baseline harness: one greedy shot (THINKING OFF), problem -> a complete Python program. No tests run,
no retry, no adaptation. This is the floor every proposed harness is measured against. (Thinking is off
here only to keep the FLOOR minimal — whether thinking helps is for the harness to decide from traces,
not a preset; the reported baseline is `react.py`, which runs with thinking on.)"""
from ..harness_base import CodeHarness
from .. import lcb_bridge as bridge

SYS = ("You are an expert competitive programmer. Read the problem and output ONE complete, self-contained "
       "Python 3 program that reads from standard input and prints the answer to standard output, inside a "
       "single ```python ... ``` block. No explanation outside the code block.")


class BareHarness(CodeHarness):
    def solve(self) -> str:
        prompt = f"{self.content}\n\nWrite the complete Python 3 solution (read stdin, print stdout)."
        resp = self.llm(prompt, system=SYS)        # thinking=False (default)
        return bridge.extract_code(resp)
