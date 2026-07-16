"""Pluggable label-free REWARD for test-time harness optimization.

The reward is the ONLY signal the group-relative selection optimizes — it must be label-free (no gold).
Implementations live behind one interface so they can be swapped (execution-health now; back-translation
consistency later, once it is debugged standalone). A reward scores one (harness, question) in [0, 1];
higher = the harness's answer looks more correct.
"""
import re
from collections import Counter
from statistics import mean

from . import bridge

SOLVE_SYS = "You are an expert Text-to-SQL system for SQLite. Output exactly one SQLite query inside ```sql``` fences."

NUM = re.compile(r"^-?\d[\d,./:%-]*$")
COUNT_W = ("how many", "number of", "count of", "total number", "how much")
LIST_W = ("which ", "list ", "name the", "names of", "what are", "give the", "find the")


def literals(sql):
    """String filter literals (skip pure numbers / very long values)."""
    return [l for l in re.findall(r"'([^']*)'", sql) if l.strip() and not NUM.match(l.strip()) and len(l) <= 60]


class Reward:
    """Interface. score() returns a label-free [0,1] estimate; score_window() averages over a set."""

    def score(self, harness, question):
        raise NotImplementedError

    def score_window(self, harness, questions):
        return mean(self.score(harness, q) for q in questions) if questions else 1.0


class ExecutionReward(Reward):
    """Execution-health composite (the current, deliberately simple signal):
        0.3 executes  +  0.3 non-empty/sane  +  0.2 filter-values-grounded  +  0.2 right cardinality.
    Builds a per-DB exact-value index ONCE for the value-grounding check. Label-free.
    """

    def __init__(self, db, per_col=300):
        self.db = db
        self.idx, self.exact = self._build_index(per_col)

    def _build_index(self, per_col):
        idx, exact = {}, set()
        for t in self.db.schema["tables"]:
            for c in t["columns"]:
                r = bridge.execute(
                    self.db,
                    f'SELECT DISTINCT "{c["name"]}" FROM "{t["name"]}" WHERE "{c["name"]}" IS NOT NULL LIMIT {per_col}',
                )
                if not r["ok"]:
                    continue
                strs = [x[0] for x in r["rows"] if isinstance(x[0], str)]
                if not (2 <= len(strs) <= 200) or max((len(s) for s in strs), default=0) > 60:
                    continue
                for v in strs:
                    idx.setdefault(v.lower().strip(), []).append(v)
                    exact.add(v)
        return idx, exact

    def score(self, harness, question):
        try:
            sql = harness.solve(question)
        except Exception:
            return 0.0
        return self.exec_from(sql, bridge.execute(self.db, sql), question)

    def exec_from(self, sql, res, question):
        """Execution-health score from a GIVEN (sql, result) — so candidates can be scored without re-solving."""
        valid = 1.0 if res["ok"] else 0.0
        sane = 1.0 if (res["ok"] and res["rows"]
                       and not (len(res["rows"]) == 1 and all(c is None for c in res["rows"][0]))) else 0.0
        lits = literals(sql)
        vg = 1.0
        if lits:
            bad = [l for l in lits if l not in self.exact and l.lower().strip() in self.idx]
            vg = 0.0 if bad else 1.0
        ql = question.lower()
        cg = 0.5
        if res["ok"]:
            if any(w in ql for w in COUNT_W):
                cg = 1.0 if (len(res["rows"]) == 1 and len(res["rows"][0]) == 1) else 0.0
            elif any(w in ql for w in LIST_W):
                cg = 1.0 if res["rows"] else 0.0
        return 0.3 * valid + 0.3 * sane + 0.2 * vg + 0.2 * cg


class SelfConsistencyReward(Reward):
    """Solve the question k times at temperature>0 (the base model) and return the plurality agreement of
    the EXECUTION results. Easy/well-posed questions yield consistent results; this correlates with
    correctness. NOTE: this is a per-QUESTION proxy (uses the base solver), not a harness discriminator."""

    def __init__(self, db, schema, k=4, temperature=0.7):
        self.db, self.schema, self.k, self.temperature = db, schema, k, temperature

    def score(self, harness, question):
        outs = bridge.solver_llm(f"Database schema:\n{self.schema}\n\nQuestion: {question}\n\nWrite the SQLite query.",
                                 system=SOLVE_SYS, temperature=self.temperature, n=self.k)
        outs = outs if isinstance(outs, list) else [outs]
        keys = []
        for o in outs:
            r = bridge.execute(self.db, bridge.extract_sql(o))
            keys.append(repr(sorted(map(repr, r["rows"]))) if r["ok"] else "ERR")
        return Counter(keys).most_common(1)[0][1] / len(keys) if keys else 0.0


class EnsembleReward(Reward):
    """Weighted mean of sub-rewards. The validated correctness PROXY is
    EnsembleReward([(1, ExecutionReward(db)), (1, SelfConsistencyReward(db, schema))]) — AUC ~0.83 (bt.py)."""

    def __init__(self, parts):
        self.parts = parts  # list of (weight, Reward)

    def score(self, harness, question):
        tot = sum(w for w, _ in self.parts) or 1.0
        return sum(w * r.score(harness, question) for w, r in self.parts) / tot
