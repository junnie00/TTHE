"""Evaluate a fixed seed harness (default: the plain ReAct baseline) on a saved LiveCodeBench slice.

Standalone baseline runner — the LCB analogue of text_to_sql/react_baseline.py. It does not invoke
harness evolution, proposers, or a judge. This produces the REFERENCE baseline: measure once, archive
the FULL evidence, never recompute.

Per problem it archives, under <out-dir>/traces/<qid>.json, the COMPLETE raw trace (every coder call
with its thinking setting, prompt and full response; every public-test run), the final code, the
public-test results and the hidden verdict — untruncated, so any later question (which problems the
baseline solves, where it fails, whether a problem has headroom) is answered from this archive instead
of by re-spending the API.

Results are cached per (harness digest, qid) so an interrupted run resumes. FAILED items are NOT
cached, so a rerun retries them — a transient error can never be frozen into the reference.

  OPENAI_API_KEY=... PYTHONPATH=. python -m livecodebench.lcb_react_baseline \
      --slice livecodebench/slices/hard60.json --harness react --run-name react_hard60 \
      --out-dir /path/to/permanent/archive
"""
import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from . import lcb_bridge as bridge
from .lcb_common import PKG_DIR, load_harness


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slice", required=True, help='json: {"items":[{"qid":..}]} or ["qid", ...]')
    ap.add_argument("--run-name", default="react_hard60")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--harness", default="react")
    ap.add_argument("--release", default="test6")
    ap.add_argument("--out-dir", default=None,
                    help="where to archive results+traces (default: logs/<run-name>, which is gitignored; "
                         "pass a path OUTSIDE the repo to keep the reference permanently)")
    ap.add_argument("--all-types", action="store_true",
                    help="include functional (LeetCode-style) problems too, not just stdin ones")
    args = ap.parse_args()

    spec = json.load(open(args.slice))
    qids = [it["qid"] for it in spec["items"]] if isinstance(spec, dict) else [
        (it["qid"] if isinstance(it, dict) else it) for it in spec]
    allp = {p.qid: p for p in bridge.load_problems(args.release, stdin_only=not args.all_types)}
    missing = [q for q in qids if q not in allp]
    if missing:
        raise ValueError(f"{len(missing)} slice qid(s) not in release {args.release}: {missing[:5]}")
    probs = [allp[q] for q in qids]

    run_dir = Path(args.out_dir) if args.out_dir else (PKG_DIR / "logs" / args.run_name)
    trace_dir = run_dir / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    cache_path = run_dir / "cache.json"
    cache = json.load(open(cache_path)) if cache_path.exists() else {}
    harness_digest = hashlib.sha1((PKG_DIR / "agents" / f"{args.harness}.py").read_bytes()).hexdigest()[:12]

    def key(qid):
        return hashlib.sha1(f"{harness_digest}\n{qid}".encode()).hexdigest()

    def run_one(position, problem):
        harness = load_harness(args.harness, problem)
        code = harness.solve()
        pub = (bridge.run_code(code, problem.public_tests, starter_code=problem.starter_code)
               if code else {"n_pass": 0, "n_total": 0, "results": []})
        correct = bool(bridge.is_correct(code, problem)) if code else False
        # ARCHIVE the complete raw evidence for this problem — untruncated, machine-readable, so later
        # analysis never needs to re-run the solver.
        (trace_dir / f"{problem.qid}.json").write_text(json.dumps({
            "qid": str(problem.qid),
            "difficulty": problem.difficulty,
            "testtype": problem.testtype,
            "starter_code": problem.starter_code,
            "harness": args.harness,
            "harness_digest": harness_digest,
            "problem": problem.content,
            "public_tests": problem.public_tests,
            "trace": harness._trace,          # every coder call (thinking, prompt, FULL response) + every run
            "final_code": code,
            "public": pub,
            "hidden_correct": correct,
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        return {
            "position": position,
            "qid": str(problem.qid),
            "difficulty": problem.difficulty,
            "testtype": problem.testtype,
            "correct": correct,
            "public_pass": pub["n_pass"],
            "public_total": pub["n_total"],
            "code_len": len(code or ""),
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
            # Cache ONLY clean results. An errored item stays uncached so a rerun retries it — a transient
            # failure must never be frozen into the reference baseline as a wrong "FAIL".
            if "error" not in record:
                cache[item_key] = record
                cache_path.write_text(json.dumps(cache, indent=2))
            completed = sum(r is not None for r in records)
            correct = sum(bool(r and r["correct"]) for r in records)
            errs = sum(bool(r and "error" in r) for r in records)
            print(f"  completed={completed}/{len(probs)} correct_so_far={correct} errors={errs}", flush=True)

    errored = [r for r in records if r and "error" in r]
    correct = sum(r["correct"] for r in records)
    result = {
        "kind": "fixed_react_baseline",
        "harness": args.harness,
        "harness_digest": harness_digest,
        "release": args.release,
        "solver_model": bridge._SOLVER_MODEL,
        "solver_base_url": bridge._cfg["base_url"],
        "max_tokens": bridge._MAX_TOKENS,
        "correct": correct,
        "total": len(records),
        "errors": len(errored),
        "errored_qids": [r["qid"] for r in errored],
        "mean_llm_calls": sum(r["llm_calls"] for r in records) / len(records),
        "mean_exec_calls": sum(r["exec_calls"] for r in records) / len(records),
        "per_problem": records,
    }
    (run_dir / "result.json").write_text(json.dumps(result, indent=2))
    print(f"\n######### REACT LCB {args.run_name} ({args.harness}): {correct}/{len(records)} = "
          f"{100 * correct / len(records):.1f}% (mean_llm_calls={result['mean_llm_calls']:.2f}) #########",
          flush=True)
    if errored:
        print(f"!!! {len(errored)} ITEM(S) ERRORED — NOT cached; rerun the same command to retry them: "
              f"{[r['qid'] for r in errored]}", flush=True)
    else:
        print(f"[clean] 0 errors. Full per-problem evidence archived at {trace_dir}/<qid>.json", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
