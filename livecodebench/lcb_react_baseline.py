"""Evaluate a fixed seed harness (default: the plain ReAct baseline) on a saved LiveCodeBench slice.

Standalone baseline runner — the LCB analogue of text_to_sql/react_baseline.py. It does not invoke
harness evolution, proposers, or a judge. Per-problem results are cached (keyed by harness digest +
qid) so an interrupted run resumes without re-spending solver calls.

  OPENAI_API_KEY=... PYTHONPATH=. python -m livecodebench.lcb_react_baseline \
      --slice livecodebench/slices/hard60.json --harness react --run-name react_hard60
"""
import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

from . import lcb_bridge as bridge
from .lcb_common import PKG_DIR, load_harness


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slice", required=True, help='json: {"items":[{"qid":..}]} or ["qid", ...]')
    ap.add_argument("--run-name", default="react_hard60")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--harness", default="react")
    ap.add_argument("--release", default="test6")
    args = ap.parse_args()

    spec = json.load(open(args.slice))
    qids = [it["qid"] for it in spec["items"]] if isinstance(spec, dict) else [
        (it["qid"] if isinstance(it, dict) else it) for it in spec]
    allp = {p.qid: p for p in bridge.load_problems(args.release, stdin_only=True)}
    missing = [q for q in qids if q not in allp]
    if missing:
        raise ValueError(f"{len(missing)} slice qid(s) not in release {args.release}: {missing[:5]}")
    probs = [allp[q] for q in qids]

    run_dir = PKG_DIR / "logs" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    cache_path = run_dir / "cache.json"
    cache = json.load(open(cache_path)) if cache_path.exists() else {}
    harness_digest = hashlib.sha1((PKG_DIR / "agents" / f"{args.harness}.py").read_bytes()).hexdigest()[:12]

    def key(qid):
        return hashlib.sha1(f"{harness_digest}\n{qid}".encode()).hexdigest()

    def run_one(position, problem):
        harness = load_harness(args.harness, problem)
        code = harness.solve()
        return {
            "position": position,
            "qid": str(problem.qid),
            "correct": bool(bridge.is_correct(code, problem)) if code else False,
            "llm_calls": sum(s["step"] == "coder_llm" for s in harness._trace),
            "exec_calls": sum(s["step"] == "run_public" for s in harness._trace),
        }

    pending, records = [], [None] * len(probs)
    for position, problem in enumerate(probs):
        item_key = key(problem.qid)
        if item_key in cache:
            records[position] = cache[item_key]
        else:
            pending.append((position, problem, item_key))

    print(f"[react-baseline] harness={args.harness} items={len(probs)} "
          f"cached={len(probs) - len(pending)} pending={len(pending)} workers={args.workers}", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_one, position, problem): (position, item_key)
                   for position, problem, item_key in pending}
        for future in as_completed(futures):
            position, item_key = futures[future]
            try:
                record = future.result()
            except Exception as error:  # noqa: BLE001
                record = {"position": position, "qid": str(probs[position].qid), "correct": False,
                          "llm_calls": 0, "exec_calls": 0, "error": f"{type(error).__name__}: {error}"}
            records[position] = record
            cache[item_key] = record
            cache_path.write_text(json.dumps(cache, indent=2))
            completed = sum(r is not None for r in records)
            correct = sum(bool(r and r["correct"]) for r in records)
            print(f"  completed={completed}/{len(probs)} correct_so_far={correct}", flush=True)

    correct = sum(r["correct"] for r in records)
    result = {
        "kind": "fixed_react_baseline",
        "harness": args.harness,
        "harness_digest": harness_digest,
        "correct": correct,
        "total": len(records),
        "mean_llm_calls": sum(r["llm_calls"] for r in records) / len(records),
        "mean_exec_calls": sum(r["exec_calls"] for r in records) / len(records),
        "per_problem": records,
    }
    (run_dir / "result.json").write_text(json.dumps(result, indent=2))
    print(f"\n######### REACT LCB {args.run_name} ({args.harness}): {correct}/{len(records)} = "
          f"{100 * correct / len(records):.1f}% (mean_llm_calls={result['mean_llm_calls']:.2f}) #########",
          flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
