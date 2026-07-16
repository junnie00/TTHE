"""LABEL-FREE evaluation of a harness on an unlabeled BIRD question stream.

Returns two things, kept strictly separate:
  - fitness   : a LABEL-FREE score (metamorphic consistency: paraphrase-INVariance + counterfactual-
                SENSitivity, the signal we measured at AUC ~0.72; combined with self-consistency when the
                harness is stochastic). This is the ONLY signal the evolution loop may use.
  - gold_acc  : execution accuracy vs gold. MEASUREMENT ONLY — never returned to the proposer / frontier.

Metamorphic is black-box (perturb the question, run the harness's own solve()), so it works for ANY
harness the proposer writes, deterministic or stochastic.
"""
import re
from statistics import mean
from concurrent.futures import ThreadPoolExecutor, as_completed

from . import bridge
from .resultkey import result_key   # canonical execution-result key (matches our prior measurements)

PARA_SYS = ("You rewrite a database question in DIFFERENT words but with EXACTLY the same meaning and the "
            "same correct answer. Do not add or drop any condition. Output ONLY the rewritten question.")
CF_SYS = ("You minimally edit a database question by changing EXACTLY ONE concrete condition to a "
          "different but valid one — a named value, a number, a superlative (highest<->lowest), or a "
          "comparison direction. Keep the structure identical; the edited question MUST have a different "
          "answer. Output ONLY the edited question.")


def _clean(text):
    line = next((l.strip() for l in (text or "").strip().splitlines() if l.strip()), "")
    return re.sub(r"^(question|rewrite|edited)\s*[:\-]?\s*", "", line, flags=re.I).strip().strip('"').strip()


def _variants(question, sys, n):
    outs = bridge.solver_llm(f"Question: {question}", system=sys, temperature=0.8, n=n)
    outs = outs if isinstance(outs, list) else [outs]
    seen, res = set(), []
    for o in outs:
        c = _clean(o)
        if c and c.lower() != question.strip().lower() and c.lower() not in seen:
            seen.add(c.lower())
            res.append(c)
    return res


def _key(harness, question):
    """Run the harness on a question, return (sql, execution key)."""
    try:
        sql = harness.solve(question)
    except Exception:                       # a buggy proposer harness shouldn't crash the eval
        return "", None
    r = bridge.execute(harness.db, sql)
    return sql, result_key(r["rows"], r["ok"])


def _metamorphic_one(harness, question, npara, ncf):
    sql0, key0 = _key(harness, question)
    if key0 is None:
        return {"sql": sql0, "key": key0, "inv": 0.0, "sens": 0.0, "meta": 0.0}
    inv = []
    for p in _variants(question, PARA_SYS, npara):
        _, k = _key(harness, p)
        inv.append(1.0 if (k is not None and k == key0) else 0.0)
    sens = []
    for c in _variants(question, CF_SYS, ncf):
        _, k = _key(harness, c)
        sens.append(1.0 if (k is not None and k != key0) else 0.0)
    invm = mean(inv) if inv else 0.0
    sensm = mean(sens) if sens else 0.0
    return {"sql": sql0, "key": key0, "inv": invm, "sens": sensm, "meta": 0.5 * invm + 0.5 * sensm}


def evaluate(harness, questions, golds=None, npara=2, ncf=2, workers=8):
    """Run the harness over `questions`. Returns dict with LABEL-FREE fitness + (if golds) gold_acc.

    `golds` (list of executed gold results) is used ONLY to compute gold_acc for measurement; it is NOT
    part of fitness. Pass golds=None during the evolution loop to guarantee no leakage.
    """
    n = len(questions)
    recs = [None] * n

    def one(i):
        m = _metamorphic_one(harness, questions[i].question, npara, ncf)
        return i, m

    with ThreadPoolExecutor(max_workers=min(workers, n)) as ex:
        for i, m in (f.result() for f in as_completed([ex.submit(one, i) for i in range(n)])):
            recs[i] = m

    for i, r in enumerate(recs):
        r["q"] = questions[i].question
    fitness = mean(r["meta"] for r in recs)
    out = {"fitness": fitness,
           "inv": mean(r["inv"] for r in recs),
           "sens": mean(r["sens"] for r in recs),
           "n": n,
           "records": recs}              # per-question (q, sql, key, inv, sens, meta) — LABEL-FREE
    if golds is not None:
        correct = []
        for i, r in enumerate(recs):
            if r["key"] is None or not golds[i]["ok"]:
                ok = 0.0
            else:
                rr = bridge.execute(harness.db, r["sql"])
                ok = 1.0 if bridge.is_correct(rr, golds[i]) else 0.0
            correct.append(ok)
            r["_gold_correct"] = ok       # MEASUREMENT ONLY — prefixed; never shown to the proposer
        out["gold_acc"] = mean(correct)
    return out
