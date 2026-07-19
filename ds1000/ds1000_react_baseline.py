"""Evaluate a fixed seed harness (default: the plain ReAct baseline) on a saved DS-1000 slice.

Standalone baseline runner — the DS-1000 analogue of text_to_sql/react_baseline.py. It does not invoke
harness evolution, proposers, or a judge. Per-problem results are cached (keyed by harness digest +
problem id) so an interrupted run resumes without re-spending solver calls.

  OPENAI_API_KEY=... PYTHONPATH=. python -m ds1000.ds1000_react_baseline \
      --ids ds1000/slices/hard50.json --harness react --run-name react_hard50
"""
import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

from . import ds1000_bridge as bridge
from .ds1000_common import PKG_DIR, load_harness


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", required=True, help="json: [problem_id, ...]")
    ap.add_argument("--run-name", default="react_hard50")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--harness", default="react")
    args = ap.parse_args()

    ids = json.load(open(args.ids))
    probs = bridge.load_problems(ids=ids)

    run_dir = PKG_DIR / "logs" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    cache_path = run_dir / "cache.json"
    cache = json.load(open(cache_path)) if cache_path.exists() else {}
    # The digest must cover the BRIDGE too, not just the harness file. react calls bridge.selfcheck, so a
    # change to the bridge changes what the baseline does while leaving agents/react.py byte-identical —
    # keying on the harness alone would silently serve results measured under the OLD bridge.
    harness_digest = hashlib.sha1(
        (PKG_DIR / "agents" / f"{args.harness}.py").read_bytes()
        + (PKG_DIR / "ds1000_bridge.py").read_bytes()
    ).hexdigest()[:12]

    def key(pid):
        return hashlib.sha1(f"{harness_digest}\n{pid}".encode()).hexdigest()

    trace_dir = run_dir / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)

    def write_trace(position, problem, code, correct, steps):
        """Persist the FULL trace: problem, every coder call, every self-check, the final code, the verdict.
        Without this the run leaves nothing to re-verify or diagnose — the score alone is not evidence."""
        L = [f"# Baseline trace — harness `{args.harness}` — Q{position}  [{problem.pid} / {problem.library}]\n",
             f"## PROBLEM\n{problem.prompt}\n",
             "## WHAT THE HARNESS DID — every coder call + every self-check, in order:"]
        for i, st in enumerate(steps, 1):
            if st.get("step") == "coder_llm":
                L.append(f"\n### step {i} — coder call (thinking={st.get('thinking')}, "
                         f"max_tokens={st.get('max_tokens')})\nPROMPT:\n{str(st.get('prompt'))}\n"
                         f"RESPONSE:\n{str(st.get('response'))}")
            else:
                L.append(f"\n### step {i} — self-check: ran={st.get('ran')}  "
                         f"redefines_input={st.get('redefines')}  error={str(st.get('error'))!r}  "
                         f"output={str(st.get('output'))!r}")
        L.append(f"\n## FINAL CODE\n```python\n{str(code)}\n```")
        L.append(f"\n## GOLD VERDICT (measurement only — never seen by the harness): "
                 f"{'CORRECT' if correct else 'WRONG'}")
        (trace_dir / f"q{position:03d}_pid{problem.pid}.md").write_text("\n".join(L), encoding="utf-8")

    def run_one(position, problem):
        harness = load_harness(args.harness, problem)
        code = harness.solve()
        correct = bool(bridge.is_correct(code, problem)) if code else False
        write_trace(position, problem, code, correct, harness._trace)
        return {
            "position": position,
            "pid": str(problem.pid),
            "correct": correct,
            "code": code,                      # persisted so the score can be INDEPENDENTLY re-verified
            "llm_calls": sum(s["step"] == "coder_llm" for s in harness._trace),
            "exec_calls": sum(s["step"] == "selfcheck" for s in harness._trace),
        }

    pending, records = [], [None] * len(probs)
    for position, problem in enumerate(probs):
        item_key = key(problem.pid)
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
                record = {"position": position, "pid": str(probs[position].pid), "correct": False,
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
    print(f"\n######### REACT DS-1000 {args.run_name} ({args.harness}): {correct}/{len(records)} = "
          f"{100 * correct / len(records):.1f}% (mean_llm_calls={result['mean_llm_calls']:.2f}) #########",
          flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
