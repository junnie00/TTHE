"""DEBUG back-translation consistency as a label-free PROXY for nl2sql correctness — variant sweep.

Signals (all label-free; gold used ONLY for AUC):
  exec      : execution-health composite (ExecutionReward)                      [reference]
  discrep   : SQL -> literal recovered question Q' -> strict 0-10 "does it answer the intended Q?"  (k judges)
  discrep_g : same, but recovery is GROUNDED in the execution result (external signal injected)
  selfcons  : solve the question k times (temp>0), plurality agreement of execution results  [reference ceiling]
  bt_exec   : mean(discrep_g, exec)                                              [ENSEMBLE — the candidate proxy]

    python -m tthe.bt --db card_games --n 50
"""
import argparse
import os
import random
import re
from collections import Counter
from statistics import mean
from concurrent.futures import ThreadPoolExecutor, wait

from . import bridge
from .agents.bare import BareHarness
from .reward import ExecutionReward

SYS = "You are an expert Text-to-SQL system for SQLite. Output exactly one SQLite query inside ```sql``` fences."


def desc(res):
    if not res["ok"]:
        return "ERROR"
    if not res["rows"]:
        return "EMPTY"
    return ("; ".join(str(r) for r in res["rows"][:3]))[:160]


def recover(schema, sql, rdesc=None):
    """MECHANICAL/literal translation — forbidden from inferring intent, so SQL bugs surface in Q'."""
    g = f"\nIt executed and returned: {rdesc}" if rdesc else ""
    prompt = (f"Schema:\n{schema}\n\nSQL:\n{sql}{g}\n\nTranslate this SQL into English MECHANICALLY — describe "
              f"ONLY what it LITERALLY does, do NOT guess intent or 'fix' anything. Map operators literally: "
              f"MAX(x)='the largest/last value of x' (NOT 'most common'); MIN(x)='the smallest value'; "
              f"GROUP BY x ORDER BY COUNT(*) DESC LIMIT 1='the most frequent x'; COUNT(*)='how many rows'; "
              f"col='v' is EXACT match, col LIKE '%v%' means 'contains v'. State the table(s)/joins, EVERY "
              f"filter literally, exactly which column(s) are returned, the exact aggregation/order/limit. "
              f"2-3 literal sentences.")
    return bridge.solver_llm(prompt, temperature=0.0).strip()


def discrep_score(schema, orig, recovered, k=3):
    """Schema-GROUNDED, literal-behaviour comparison — no hallucinated tables, concrete mismatches only."""
    prompt = (f"Database schema (the ONLY tables/columns that exist — do NOT invent others):\n{schema}\n\n"
              f"INTENDED question: {orig}\n\nWhat the SQL LITERALLY does: {recovered}\n\nDoes the SQL's literal "
              f"behaviour correctly answer the intended question? Flag a mismatch ONLY for a CONCRETE "
              f"difference in (a) which column/entity is returned, (b) a filter condition, or (c) the "
              f"aggregation (e.g. wants 'most common' but SQL takes MAX; wants names but returns ids; "
              f"wrong/missing filter). Do NOT speculate about tables/columns/data not shown. On the LAST line "
              f"output a single integer 0-10 (10 = literal behaviour exactly answers the question).")
    outs = bridge.solver_llm(prompt, temperature=0.7, n=k)
    outs = outs if isinstance(outs, list) else [outs]
    vals = [min(int(re.findall(r"\b(\d{1,2})\b", r)[-1]), 10) / 10.0 for r in outs if re.findall(r"\b(\d{1,2})\b", r)]
    return mean(vals) if vals else 0.5


def projection_score(schema, question, sql, k=3):
    """Judge ONLY the OUTPUT PROJECTION (the SELECT list), ignoring filters/joins/values. The single most
    common 'computed-right-but-wrong-shape' error: the SQL returns extra columns, drops the DISTINCT a
    single-value question needs, or selects the wrong attribute. Visible directly in SQL+question, so far
    more verifiable than the result VALUE. 0-10."""
    prompt = (f"Schema (only these tables/columns exist):\n{schema}\n\nQUESTION: {question}\n\nSQL:\n{sql}\n\n"
              f"Judge ONLY the SELECT/output columns — IGNORE the filters, joins and the data values. Does "
              f"the SQL return EXACTLY the attribute(s) the question asks for? Flag a mismatch for: "
              f"(a) EXTRA columns the question did not ask for, (b) a MISSING asked attribute, (c) a missing "
              f"DISTINCT when the question expects a single/unique value or list. On the LAST line output one "
              f"integer 0-10 (10 = the projection returns exactly what was asked, nothing extra/missing).")
    outs = bridge.solver_llm(prompt, temperature=0.5, n=k)
    outs = outs if isinstance(outs, list) else [outs]
    vals = [min(int(re.findall(r"\b(\d{1,2})\b", r)[-1]), 10) / 10.0 for r in outs if re.findall(r"\b(\d{1,2})\b", r)]
    return mean(vals) if vals else 0.5


def struct_extract(schema, text, is_sql):
    head = "SQL" if is_sql else "Question"
    what = "this SQL literally computes" if is_sql else "this question asks for"
    prompt = (f"Schema:\n{schema}\n\n{head}:\n{text}\n\nState what {what}, as exactly 4 lines:\n"
              f"TARGET: <what is selected/counted>\nFILTERS: <every condition as column=value, or none>\n"
              f"AGG: <count/sum/avg/max/min/none>\nGROUP/ORDER/LIMIT: <or none>")
    return bridge.solver_llm(prompt, temperature=0.0)


def struct_match(s_sql, s_q, k=2):
    prompt = (f"A) what a SQL computes:\n{s_sql}\n\nB) what the question asks for:\n{s_q}\n\n"
              f"For EACH field (TARGET, FILTERS, AGG, GROUP/ORDER/LIMIT) say MATCH or MISMATCH — A must "
              f"satisfy B exactly (right target, exact filters, right aggregation). On the LAST line output "
              f"a single integer 0-4 = how many of the 4 fields MATCH.")
    outs = bridge.solver_llm(prompt, temperature=0.5, n=k)
    outs = outs if isinstance(outs, list) else [outs]
    vals = [int(re.findall(r"\b([0-4])\b", r)[-1]) / 4.0 for r in outs if re.findall(r"\b([0-4])\b", r)]
    return mean(vals) if vals else 0.5


def selfcons_score(schema, db, question, k=4):
    outs = bridge.solver_llm(f"Database schema:\n{schema}\n\nQuestion: {question}\n\nWrite the SQLite query.",
                             system=SYS, temperature=0.7, n=k)
    outs = outs if isinstance(outs, list) else [outs]
    keys = []
    for o in outs:
        r = bridge.execute(db, bridge.extract_sql(o))
        keys.append(repr(sorted(map(repr, r["rows"]))) if r["ok"] else "ERR")
    return Counter(keys).most_common(1)[0][1] / len(keys) if keys else 0.0


def auc(scores, labels):
    pos = [s for s, l in zip(scores, labels) if l]
    neg = [s for s, l in zip(scores, labels) if not l]
    if not pos or not neg:
        return float("nan")
    return sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in pos for n in neg) / (len(pos) * len(neg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="card_games")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--shuffle", type=int, default=7)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    db = bridge.get_db(args.db)
    qs = bridge.eval_questions(args.db)
    order = list(range(len(qs)))
    random.Random(args.shuffle).shuffle(order)
    stream = [qs[i] for i in order[:args.n]]
    golds = [bridge.gold_result(db, q.gold_sql) for q in stream]
    bare = BareHarness(db)
    schema = bare.schema
    print(f"[bt] building ExecutionReward index ...", flush=True)
    exe = ExecutionReward(db)
    print(f"[bt] {args.db}: scoring {len(stream)} questions ...", flush=True)
    done = [0]

    def one(i):
        q = stream[i]
        try:
            sql = bare.solve(q.question)
        except Exception:
            sql = ""
        res = bridge.execute(db, sql) if sql else {"ok": False, "rows": []}
        correct = bool(res["ok"] and golds[i]["ok"] and bridge.compare_results(res["rows"], golds[i]["rows"]))
        ex = exe.score(bare, q.question)
        disc = discrep_score(schema, q.question, recover(schema, sql)) if sql else 0.0
        sc = selfcons_score(schema, db, q.question)
        done[0] += 1
        print(f"  [{done[0]}/{len(stream)}] correct={int(correct)} exec={ex:.2f} discrep={disc:.2f} "
              f"selfcons={sc:.2f}", flush=True)
        return i, {"correct": correct, "exec": ex, "discrep": disc, "selfcons": sc,
                   "bt_exec": (disc + ex) / 2, "bt_all": (ex + disc + sc) / 3}

    recs = [None] * len(stream)
    ex_pool = ThreadPoolExecutor(max_workers=args.workers)
    futs = {ex_pool.submit(one, i): i for i in range(len(stream))}
    done, not_done = wait(futs, timeout=700)             # per-batch deadline; drop hung questions
    for f in done:
        try:
            i, r = f.result()
            recs[i] = r
        except Exception:
            pass
    recs = [r for r in recs if r is not None]

    labels = [r["correct"] for r in recs]
    base = mean(1.0 if l else 0.0 for l in labels)
    print(f"\n########## label-free proxies for correctness (N={len(recs)}, bare acc={base:.3f}) ##########")
    for sig in ("exec", "discrep", "selfcons", "bt_exec", "bt_all"):
        print(f"  {sig:10s} AUC = {auc([r[sig] for r in recs], labels):.3f}")
    os._exit(0)                                          # don't block on hung query threads


if __name__ == "__main__":
    main()
