"""TEST-TIME harness optimization for LiveCodeBench (code domain) — the live loop (mirror of
text_to_sql/optimize.py, agentic batch generate->judge).

ONE general CodeHarness (arbitrary Python) starts from `bare` (thinking OFF) and ACCUMULATES across
batches. Per batch: OBSERVE (run each candidate, write a full trace = problem + every coder call (+thinking
choice) + final code + PUBLIC-test results) -> GENERATE (G agentic generators deep-read all traces + write
an improved harness) -> PICK (one agentic judge picks the harness passing the most PUBLIC tests) -> SCORE
that batch with the chosen harness on HIDDEN tests (MEASUREMENT ONLY). The label-free signal is PUBLIC
tests; hidden tests never enter the loop. Bare baseline cached per-qid in logs/bare_cache.json.

    cd <repo-root> && ANTHROPIC_BASE_URL=https://api.deepseek.com/anthropic ANTHROPIC_AUTH_TOKEN=... \
      TTHO_PROPOSER_MODEL=deepseek-v4-flash OPENAI_API_KEY=... PYTHONPATH=. \
      python -u -m livecodebench.lcb_optimize --pilot livecodebench/logs/pilot50.json \
      --batch-size 5 --group 2 --max-rounds 3 --run-name pilot50
"""
import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor

from . import lcb_bridge as bridge
from . import lcb_proposer as P
from .lcb_common import load_harness, PKG_DIR, AGENTS_DIR

_SOLVE_POOL = ThreadPoolExecutor(max_workers=32)


def safe_solve(h, timeout):
    """Run a (proposer-written) harness's solve() under a hard wall-clock cap so a buggy harness can't hang."""
    try:
        return _SOLVE_POOL.submit(h.solve).result(timeout=timeout) or ""
    except Exception:
        return ""


def _loadable(name, problem):
    try:
        load_harness(name, problem)
        return True
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", required=True, help="json with {items:[{qid,difficulty}...]}")
    ap.add_argument("--version", default="test6")
    ap.add_argument("--group", type=int, default=2, help="G agentic generators per GENERATE round")
    ap.add_argument("--max-rounds", type=int, default=3, help="GENERATE rounds per batch")
    ap.add_argument("--batch-size", type=int, default=5)
    ap.add_argument("--propose-timeout", type=int, default=600, help="hard cap per generator/judge claude session")
    ap.add_argument("--solve-timeout", type=int, default=900, help="hard cap per harness.solve (anti-hang)")
    ap.add_argument("--model", default=os.environ.get("TTHO_PROPOSER_MODEL", "deepseek-v4-flash"))
    ap.add_argument("--run-name", default="lcbpilot")
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--initial-harness", default="bare",
                    help="seed harness the evolution starts from (e.g. react)")
    args = ap.parse_args()

    if args.fresh:
        for f in AGENTS_DIR.glob("cand_*.py"):   # only clear generated candidates; keep seed harnesses
            f.unlink()

    allp = {p.qid: p for p in bridge.load_problems(args.version, stdin_only=True)}
    spec = json.load(open(args.pilot))["items"]
    items = []
    for it in spec:
        p = allp[it["qid"]]
        p.difficulty = it["difficulty"]
        items.append(p)

    run_dir = PKG_DIR / "logs" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    log = open(run_dir / "opt_log.jsonl", "w")
    print(f"\n######### TEST-TIME harness optimization — LiveCodeBench (agentic, public-test signal) #########")
    diffs = {}
    for p in items:
        diffs[p.difficulty] = diffs.get(p.difficulty, 0) + 1
    print(f"[stream] {len(items)} problems  by difficulty={diffs}  batch_size={args.batch_size} "
          f"group={args.group} rounds={args.max_rounds}", flush=True)

    def write_trace(trace_dir, name, j, problem, code, pub, steps):
        L = [f"# Trace — harness `{name}` — Q{j}  [{problem.qid} / {problem.difficulty}]\n",
             f"## PROBLEM\n{problem.content}\n",
             "## WHAT THE HARNESS DID — every coder call + every public-test run, in order:"]
        for i, st in enumerate(steps, 1):
            if st.get("step") == "coder_llm":
                L.append(f"\n### step {i} — coder call (thinking={st.get('thinking')})\nPROMPT:\n{str(st.get('prompt'))[:1200]}\n"
                         f"RESPONSE:\n{str(st.get('response'))[:4000]}")
            else:
                L.append(f"\n### step {i} — public-test run: {st.get('n_pass')}/{st.get('n_total')}")
        # FINAL CODE must be COMPLETE — it is what the judge re-runs and what the proposer diagnoses. A cap
        # here truncated long solutions mid-statement, so the proposer misread truncation as a syntax bug and
        # "fixed" a phantom; never truncate the final artifact.
        L.append(f"\n## FINAL CODE\n```python\n{str(code)}\n```")
        L.append(f"\n## PUBLIC-TEST RESULTS ({pub['n_pass']}/{pub['n_total']} passed) — the label-free signal:")
        for k, r in enumerate(pub.get("results", [])[:6]):
            L.append(f"  test{k}: {'PASS' if r['ok'] else 'FAIL'}  input={r['input']!r}  expected={r['expected']!r}  got={r['stdout']!r}  err={r['stderr']!r}")
        L.append(f"\n## BACK-TRANSLATION — what the FINAL CODE literally does, in plain English. COMPARE it to "
                 f"the PROBLEM above: if it describes something different from what the problem asks, the code "
                 f"is likely wrong (an intent-level check beyond the public tests).\n{bridge.back_translate(code)}")
        (trace_dir / f"{name}__q{j}.md").write_text("\n".join(L), encoding="utf-8")

    def observe(name, batch, trace_dir):
        """Run harness `name` on every batch problem (parallel); write each trace; return list of codes."""
        cls = type(load_harness(name, batch[0]))          # reload + class ONCE (reload not thread-safe)
        codes = [None] * len(batch)

        def one(jp):
            j, p = jp
            h = cls(p)
            code = safe_solve(h, args.solve_timeout)
            pub = bridge.run_code(code, p.public_tests) if code else {"n_pass": 0, "n_total": len(p.public_tests), "results": []}
            write_trace(trace_dir, name, j, p, code, pub, getattr(h, "_trace", []))
            codes[j] = code
        with ThreadPoolExecutor(max_workers=min(len(batch), 8)) as ex:
            list(ex.map(one, list(enumerate(batch))))
        return codes

    H = args.initial_harness
    if not (AGENTS_DIR / f"{H}.py").exists():
        raise ValueError(f"--initial-harness not found: agents/{H}.py")
    B = args.batch_size
    batches = [items[i:i + B] for i in range(0, len(items), B)]
    tt_correct, tt_total, tt_log, ev_results = 0, 0, [], []
    for bi, batch in enumerate(batches):
        trace_dir = run_dir / "traces" / f"b{bi}"
        trace_dir.mkdir(parents=True, exist_ok=True)
        traced, cand_results = set(), {}
        branches = [H] * args.group
        print(f"\n===== BATCH {bi}/{len(batches)} ({len(batch)} q) — start from H={H} =====", flush=True)
        # GENERATE phase: all branches share peer evidence, but each proposer edits only its assigned base.
        # A failed/unloadable child leaves that branch at its previous parent.
        for rnd in range(args.max_rounds):
            active = list(dict.fromkeys(branches))
            for c in active:
                if c not in traced and _loadable(c, batch[0]):
                    cand_results[c] = observe(c, batch, trace_dir)
                    traced.add(c)
            proposed = P.sample_branches(branches, trace_dir, run_dir, f"b{bi}r{rnd}", args.run_name,
                                         batch, args.model, args.propose_timeout)
            next_branches, advanced = [], 0
            for base, child in zip(branches, proposed):
                accepted = child if child and _loadable(child, batch[0]) else base
                next_branches.append(accepted)
                advanced += accepted != base
            branches = next_branches
            for c in dict.fromkeys(branches):
                if c not in traced and _loadable(c, batch[0]):
                    cand_results[c] = observe(c, batch, trace_dir)
                    traced.add(c)
            print(f"   batch{bi} gen-round{rnd}: {advanced}/{args.group} branches advanced", flush=True)
        # PICK phase: the judge sees ONLY the final active branches, never the historical archive.
        final = list(dict.fromkeys(branches))
        for c in final:
            if c not in traced and _loadable(c, batch[0]):
                cand_results[c] = observe(c, batch, trace_dir)
                traced.add(c)
        picked = P.pick_batch(final, trace_dir, run_dir, f"b{bi}", args.model, args.propose_timeout)
        H = picked if picked in final else final[0]
        print(f"   batch{bi}: final branches={branches} ({len(final)} unique) -> JUDGE picked H={H}", flush=True)
        codes = cand_results.get(H)
        if codes is not None:
            bc = 0
            for code, p in zip(codes, batch):
                ok = bool(bridge.is_correct(code, p))
                ev_results.append({"qid": p.qid, "difficulty": p.difficulty, "correct": ok, "harness": H})
                bc += ok
            tt_correct += bc
            tt_total += len(batch)
            tt_log.append({"batch": bi, "harness": H, "correct": bc, "total": len(batch)})
            print(f"   [test-time] batch{bi} H={H}: {bc}/{len(batch)} (hidden tests)", flush=True)
        log.write(json.dumps({"batch": bi, "harness": H, "branches": branches, "candidates": final}) + "\n")
        log.flush()
    log.close()

    print(f"\n######### RESULT (test-time / transductive) — final H = {H} #########", flush=True)
    print(f"  test-time evolved = {tt_correct}/{tt_total}   (baseline = plain react, measured separately)")
    print("  by difficulty (evolved):")
    for diff in sorted(diffs):
        ev_d = sum(r["correct"] for r in ev_results if r["difficulty"] == diff)
        ev_n = sum(1 for r in ev_results if r["difficulty"] == diff)
        print(f"    {diff:8} evolved {ev_d}/{ev_n}")
    json.dump({"tt_correct": tt_correct, "tt_total": tt_total, "final_harness": H,
               "batches": tt_log, "per_problem": ev_results},
              open(run_dir / "result.json", "w"), indent=2)
    print(f"[saved] {run_dir}/result.json   [traces] {run_dir}/traces/")


if __name__ == "__main__":
    main()
