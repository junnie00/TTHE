"""Baseline harness: one greedy shot, schema + question -> SQL. No retrieval, no voting, no adaptation.
This is the floor every proposed harness is measured against."""
from ..harness_base import SQLHarness
from .. import bridge

SYS = ("You are an expert Text-to-SQL system for SQLite. Read the schema carefully and output exactly "
       "one SQLite query that answers the question, inside a ```sql ... ``` block.")


class BareHarness(SQLHarness):
    def solve(self, question: str) -> str:
        prompt = f"Database schema:\n{self.schema}\n\nQuestion: {question}\n\nWrite the SQLite query."
        resp = self.llm(prompt, system=SYS, temperature=0.0)
        return bridge.extract_sql(resp)
