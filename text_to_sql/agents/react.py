"""Plain ReAct Text-to-SQL harness — the execution-retry baseline.

The first model call is IDENTICAL to the single-shot bare harness (schema +
question). If the generated SQL executes without a database error it is
submitted immediately; only a SQLite execution error triggers another attempt,
with the previous SQL and the error appended. There is no proactive probing, no
task-specific repair guidance, and no checking of whether the returned rows are
correct — those are behaviours the harness evolution is meant to discover, not
part of the baseline.
"""
from ..harness_base import SQLHarness
from .. import bridge


MAX_ATTEMPTS = 3

SYS = ("You are an expert Text-to-SQL system for SQLite. Read the schema carefully and output exactly "
       "one SQLite query that answers the question, inside a ```sql ... ``` block.")


class ReactHarness(SQLHarness):
    def solve(self, question: str) -> str:
        history = []
        last_sql = ""
        for _ in range(MAX_ATTEMPTS):
            prompt = f"Database schema:\n{self.schema}\n\nQuestion: {question}\n\nWrite the SQLite query."
            if history:
                prompt += "\n\nPrevious failed attempts:\n" + "\n\n".join(history)
            resp = self.llm(prompt, system=SYS, temperature=0.0)
            sql = bridge.extract_sql(resp).strip()
            last_sql = sql
            if not sql:
                history.append("SQL: (none)\nDatabase error: no SQL query was produced")
                continue
            result = self.execute(sql)
            if result.get("ok"):                       # executes without error -> submit (rows not checked)
                return sql
            history.append(f"SQL:\n{sql}\nDatabase error:\n{str(result.get('error', 'unknown SQLite error'))[:1200]}")
        return last_sql
