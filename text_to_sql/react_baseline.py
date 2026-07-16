"""Evaluate the fixed generic ReAct harness on a saved BIRD slice.

This is a standalone baseline runner. It does not invoke harness evolution,
generators, or a judge, and it does not modify the existing optimizer.
Per-question results are cached so interrupted runs can resume without
re-spending solver calls.
"""

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from . import bridge
from .evolve import PKG_DIR, load_harness


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cross-set", required=True)
    ap.add_argument("--run-name", default="react_mini50")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--harness", default="react")
    args = ap.parse_args()

    cross_path = Path(args.cross_set)
    spec = json.load(open(cross_path))["cross"]
    items = [(db_id, bridge.eval_questions(db_id)[idx]) for db_id, idx in spec]
    details_path = cross_path.with_name(f"{cross_path.stem}_details.json")
    if details_path.exists():
        expected = json.loads(details_path.read_text(encoding="utf-8"))
        if len(expected) != len(items):
            raise ValueError(
                f"cross-set identity mismatch: {len(items)} indexed items but "
                f"{len(expected)} detail records in {details_path}"
            )
        mismatches = []
        for position, ((db_id, question), detail) in enumerate(zip(items, expected)):
            actual_text = question.question.partition("\nHint:")[0].strip()
            expected_text = detail["question"].strip()
            if db_id != detail["db_id"] or actual_text != expected_text:
                mismatches.append(
                    (position, db_id, actual_text, detail["db_id"], expected_text)
                )
        if mismatches:
            position, db_id, actual_text, expected_db, expected_text = mismatches[0]
            raise ValueError(
                "cross-set dataset identity mismatch "
                f"({len(mismatches)}/{len(items)} items; first at position {position}): "
                f"loaded [{db_id}] {actual_text!r}, expected "
                f"[{expected_db}] {expected_text!r}. "
                "Set BIRD_DEV_FILE to the dataset file used to build the cross-set."
            )
    run_dir = PKG_DIR / "logs" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    cache_path = run_dir / "cache.json"
    cache = json.load(open(cache_path)) if cache_path.exists() else {}
    harness_path = PKG_DIR / "agents" / f"{args.harness}.py"
    harness_digest = hashlib.sha1(harness_path.read_bytes()).hexdigest()[:12]

    def key(db_id, question):
        payload = f"{harness_digest}\n{db_id}\n{question}".encode()
        return hashlib.sha1(payload).hexdigest()

    def run_one(position, db_id, question):
        db = bridge.get_db(db_id)
        harness = load_harness(args.harness, db)
        sql = harness.solve(question.question)
        result = bridge.execute(db, sql) if sql else {"ok": False, "rows": []}
        gold = bridge.gold_result(db, question.gold_sql)
        return {
            "position": position,
            "db_id": db_id,
            "correct": bool(bridge.is_correct(result, gold)),
            "sql": sql,
            "llm_calls": sum(step["step"] == "coder_llm" for step in harness._trace),
            "exec_calls": sum(step["step"] == "execute_sql" for step in harness._trace),
        }

    pending = []
    records = [None] * len(items)
    for position, (db_id, question) in enumerate(items):
        item_key = key(db_id, question.question)
        if item_key in cache:
            records[position] = cache[item_key]
        else:
            pending.append((position, db_id, question, item_key))

    print(
        f"[react-baseline] items={len(items)} cached={len(items) - len(pending)} "
        f"pending={len(pending)} workers={args.workers}",
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(run_one, position, db_id, question): (position, item_key)
            for position, db_id, question, item_key in pending
        }
        for future in as_completed(futures):
            position, item_key = futures[future]
            try:
                record = future.result()
            except Exception as error:
                record = {
                    "position": position,
                    "db_id": items[position][0],
                    "correct": False,
                    "sql": "",
                    "llm_calls": 0,
                    "exec_calls": 0,
                    "error": f"{type(error).__name__}: {error}",
                }
            records[position] = record
            cache[item_key] = record
            cache_path.write_text(json.dumps(cache, indent=2))
            completed = sum(record is not None for record in records)
            correct = sum(bool(record and record["correct"]) for record in records)
            print(f"  completed={completed}/{len(items)} correct_so_far={correct}", flush=True)

    correct = sum(record["correct"] for record in records)
    result = {
        "kind": "fixed_execution_retry_baseline",
        "harness": args.harness,
        "harness_digest": harness_digest,
        "correct": correct,
        "total": len(records),
        "mean_llm_calls": sum(record["llm_calls"] for record in records) / len(records),
        "mean_exec_calls": sum(record["exec_calls"] for record in records) / len(records),
        "per_problem": records,
    }
    (run_dir / "result.json").write_text(json.dumps(result, indent=2))
    print(
        f"[result] react={correct}/{len(records)} "
        f"mean_llm_calls={result['mean_llm_calls']:.2f} "
        f"mean_exec_calls={result['mean_exec_calls']:.2f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
