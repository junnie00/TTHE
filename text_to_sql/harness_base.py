"""The HARNESS interface (analogue of meta-harness's MemorySystem, for Text-to-SQL).

A harness is ARBITRARY PYTHON wrapping the FROZEN weak solver. Unlike a prompt + a fixed menu of
toggles, `solve()` can do anything the proposer writes: retrieve real DB values, decompose into
sub-steps, self-verify against the schema, repair failed queries, etc. (Consensus across samples is a
WEAK prior, not proof — it agrees with the truth on easy questions but can be confidently wrong on hard
ones where the model's errors are correlated; prefer objective signals like execution results, use
agreement only as a soft tiebreaker.) This is the meta-harness representation — the proposer writes
subclasses of SQLHarness.

LABEL-FREE: a harness may adapt ONLINE on the unlabeled test stream via observe(), which receives only
the question, its own SQL, and the EXECUTION result — never the gold. Gold is used outside, for
measurement only.
"""
from abc import ABC, abstractmethod

from . import bridge


class SQLHarness(ABC):
    """Subclass this. You may rewrite ANY part of a harness — control flow, prompts, number of solver
    calls, self-verification, voting, repair loops — and `import bridge` to override any call-layer
    parameter. Your action space is the whole Python file, NOT just the helpers below. Two invariants are
    fixed (audit_harness.py checks them; a candidate that breaks either is invalid):

      * FROZEN SOLVER — never change WHO answers: no new client, no reassigning bridge._client /
        _SOLVER_MODEL, no base_url / api_key, no importing a network library. You MAY freely change HOW
        you call it (prompt, temperature, number of calls, voting).
      * LABEL-FREE — read only label-free signals (schema, execution results). Never read gold: a
        question's gold_sql / gold_result, or bridge.is_correct / any grading answer.

    Convenience helpers (use or ignore):
        self.db          — the database (has .schema_text(), .execute())
        self.schema      — db.schema_text() (string, precomputed)
        self.llm(prompt, system="", temperature=0.0, n=1)  — the frozen solver
        self.execute(sql)  -> {ok, rows, ...}              — run SQL on self.db
        bridge.extract_sql(text) -> str                    — pull SQL out of an LLM response
    solve() must return the final SQL string.
    """

    def __init__(self, db):
        self.db = db
        self.schema = db.schema_text()
        self._trace = []          # FULL execution trace of this solve: every coder call + every SQL run, in order

    # convenience wrappers (so harness code reads cleanly) — they also RECORD into self._trace so the
    # proposer can deep-read the harness's complete step-by-step behaviour (not a compressed summary).
    def llm(self, prompt, system="", temperature=0.0, n=1):
        out = bridge.solver_llm(prompt, system=system, temperature=temperature, n=n)
        self._trace.append({"step": "coder_llm", "system": system, "prompt": prompt,
                            "response": out if isinstance(out, str) else list(out)})
        return out

    def execute(self, sql):
        res = bridge.execute(self.db, sql)
        self._trace.append({"step": "execute_sql", "sql": sql, "ok": res.get("ok"),
                            "error": res.get("error"), "rows": res.get("rows", [])[:5], "n_rows": len(res.get("rows", []))})
        return res

    def tables(self):
        """{table_name: [column_name, ...]} straight from the live DB. Use THIS — do NOT regex-parse
        self.schema (its format is `Table name(col1, col2, ...)`, not CREATE TABLE)."""
        return {t["name"]: [c["name"] for c in t["columns"]] for t in self.db.schema["tables"]}

    def column_types(self, table):
        """{column_name: type_string} for one table."""
        for t in self.db.schema["tables"]:
            if t["name"] == table:
                return {c["name"]: c.get("type", "") for c in t["columns"]}
        return {}

    def distinct(self, table, column, limit=50):
        """Distinct stored values of a column (exact strings) — for matching question literals to the
        real value. Returns [] on error."""
        res = self.execute(f'SELECT DISTINCT "{column}" FROM "{table}" WHERE "{column}" IS NOT NULL LIMIT {limit}')
        return [r[0] for r in res["rows"]] if res["ok"] else []

    @abstractmethod
    def solve(self, question: str) -> str:
        """Return a single SQLite query string answering `question` over self.db."""
        ...

    # ---- optional LABEL-FREE online adaptation on the test stream (NO gold, ever) ----
    def observe(self, question: str, sql: str, exec_result: dict) -> None:
        """Called after solve() with the execution result (no gold). Default: no adaptation."""
        return None

    def get_state(self) -> str:
        return ""

    def set_state(self, state: str) -> None:
        return None
