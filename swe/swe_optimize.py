"""TEST-TIME harness optimization for SWE-bench Verified — the live loop (mirror of livecodebench/lcb_optimize.py).

ONE general SWEHarness (arbitrary Python wrapping a FROZEN mini-swe-agent rollout) starts from `bare` and
ACCUMULATES across batches. Per batch: OBSERVE (run each candidate on the batch, write a full trace = issue +
agent trajectory + final patch) -> GENERATE (G agentic generators deep-read all traces + write an improved
harness) -> PICK (one agentic judge picks the best harness from label-free trace evidence) -> SCORE that
batch with the chosen harness via the official swebench harness (MEASUREMENT ONLY). The label-free signal is
the agent's own in-container execution; gold hidden tests never enter the loop. Bare baseline cached per
instance_id in logs/bare_cache.json.

    cd <monorepo root> && ANTHROPIC_BASE_URL=https://api.deepseek.com/anthropic ANTHROPIC_AUTH_TOKEN=... \
      TTHO_PROPOSER_MODEL=deepseek-v4-flash OPENAI_API_KEY=... PYTHONPATH=. \
      python -u -m swebench.swe_optimize --pilot swebench/logs/pilot30_ids.json \
      --batch-size 5 --group 2 --max-rounds 2 --run-name pilot30
"""
import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor

from . import swe_bridge as bridge
from . import swe_proposer as P
from .swe_common import load_harness, PKG_DIR, AGENTS_DIR

_SOLVE_POOL = ThreadPoolExecutor(max_workers=32)


def safe_solve(h, timeout):
    """Run a (proposer-written) harness's solve() under a hard wall-clock cap so a buggy harness can't hang.
    Always tears down the harness's Docker container afterwards (the harness now owns the env lifecycle)."""
    try:
        return _SOLVE_POOL.submit(h.solve).result(timeout=timeout) or ""
    except Exception:
        return ""
    finally:
        try:
            h.cleanup()
        except Exception:
            pass


def _loadable(name, instance):
    try:
        load_harness(name, instance)
        return True
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", required=True, help="json: list of instance_ids OR {items:[{instance_id}...]}")
    ap.add_argument("--group", type=int, default=2, help="G agentic generators per GENERATE round")
    ap.add_argument("--max-rounds", type=int, default=2, help="GENERATE rounds per batch")
    ap.add_argument("--batch-size", type=int, default=5)
    ap.add_argument("--propose-timeout", type=int, default=900, help="hard cap per generator/judge claude session")
    ap.add_argument("--solve-timeout", type=int, default=1800, help="hard cap per harness.solve (a rollout is slow)")
    ap.add_argument("--model", default=os.environ.get("TTHO_PROPOSER_MODEL", "deepseek-v4-flash"))
    ap.add_argument("--run-name", default="swepilot")
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--initial-harness", default="bare",
                    help="seed harness the evolution starts from (e.g. react)")
    args = ap.parse_args()

    if args.fresh:
        for f in AGENTS_DIR.glob("cand_*.py"):   # only clear generated candidates; keep seed harnesses
            f.unlink()

    spec = json.load(open(args.pilot))
    ids = spec["items"] if isinstance(spec, dict) else spec
    ids = [it["instance_id"] if isinstance(it, dict) else it for it in ids]
    items = bridge.load_instances(ids=ids)

    run_dir = PKG_DIR / "logs" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    log = open(run_dir / "opt_log.jsonl", "w")
    print(f"\n######### TEST-TIME harness optimization — SWE-bench Verified (agentic, label-free signal) #########")
    repos = {}
    for inst in items:
        repos[inst["repo"]] = repos.get(inst["repo"], 0) + 1
    print(f"[stream] {len(items)} instances  by repo={repos}  batch_size={args.batch_size} "
          f"group={args.group} rounds={args.max_rounds}", flush=True)

    def write_trace(trace_dir, name, j, instance, patch, steps):
        iid = instance["instance_id"]
        L = [f"# Trace — harness `{name}` — I{j}  [{iid} / {instance['repo']}]\n",
             f"## ISSUE\n{instance['problem_statement'][:4000]}\n",
             "## WHAT THE AGENT DID — its trajectory (reproduction scripts + test runs = the label-free evidence):"]
        msgs = []
        for st in steps:
            if st.get("step") == "agent_rollout":
                msgs += st.get("messages", [])
        shown = 0
        for m in msgs:
            if shown >= 15:
                L.append("\n  ... (trajectory truncated)")
                break
            role = m.get("role")
            if role == "system":
                continue
            content = str(m.get("content", ""))
            if role == "assistant":
                L.append(f"\n### agent step {shown+1}\n{content[:800]}")
            else:
                L.append(f"\nOBSERVATION:\n{content[:800]}")
            shown += 1
        L.append(f"\n## FINAL PATCH\n```diff\n{str(patch)}\n```")   # never truncate the final artifact the judge evaluates
        L.append("\nNOTE: a NON-EMPTY, targeted patch whose trajectory shows the agent REPRODUCED the issue and "
                 "VERIFIED the fix (and only touches source, not tests) is the label-free signal — the gold hidden "
                 "tests are NEVER shown here.")
        (trace_dir / f"{name}__i{j}.md").write_text("\n".join(L), encoding="utf-8")

    def observe(name, batch, trace_dir):
        """Run harness `name` on every batch instance (parallel, Docker rollouts are slow); write each trace;
        return list of patches."""
        cls = type(load_harness(name, batch[0]))          # reload + class ONCE (reload not thread-safe)
        patches = [None] * len(batch)

        def one(ji):
            j, inst = ji
            h = cls(inst)
            patch = safe_solve(h, args.solve_timeout)
            write_trace(trace_dir, name, j, inst, patch, getattr(h, "_trace", []))
            patches[j] = patch
        with ThreadPoolExecutor(max_workers=min(len(batch), 5)) as ex:
            list(ex.map(one, list(enumerate(batch))))
        return patches

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
        branches = [H] * args.group                                    # G fixed branches, each seeded from H
        print(f"\n===== BATCH {bi}/{len(batches)} ({len(batch)} i) — start from H={H} =====", flush=True)
        # GENERATE phase: all branches share peer evidence, but proposer gi edits only branch gi. A failed
        # child leaves only that branch at its previous parent.
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
                if child and _loadable(child, batch[0]):
                    next_branches.append(child)
                    advanced += 1
                else:
                    next_branches.append(base)
            branches = next_branches
            for c in dict.fromkeys(branches):
                if c not in traced and _loadable(c, batch[0]):
                    cand_results[c] = observe(c, batch, trace_dir)
                    traced.add(c)
            print(f"   batch{bi} gen-round{rnd}: {advanced}/{args.group} branches advanced", flush=True)
        # PICK phase: the judge sees only the final active branches, never the historical archive.
        final = list(dict.fromkeys(branches))
        for c in final:
            if c not in traced and _loadable(c, batch[0]):
                cand_results[c] = observe(c, batch, trace_dir)
                traced.add(c)
        H = P.pick_batch(final, trace_dir, run_dir, f"b{bi}", args.model, args.propose_timeout) or H
        print(f"   batch{bi}: final branches={branches} ({len(final)} unique) -> JUDGE picked H={H}", flush=True)
        patches = cand_results.get(H)
        if patches is not None:
            gold = bridge.is_correct_batch(list(zip(batch, patches)), run_id=f"{args.run_name}_b{bi}_eval")
            bc = 0
            for inst in batch:
                ok = bool(gold.get(inst["instance_id"], False))
                ev_results.append({"instance_id": inst["instance_id"], "repo": inst["repo"], "correct": ok, "harness": H})
                bc += ok
            tt_correct += bc
            tt_total += len(batch)
            tt_log.append({"batch": bi, "harness": H, "correct": bc, "total": len(batch)})
            print(f"   [test-time] batch{bi} H={H}: {bc}/{len(batch)} (gold resolved)", flush=True)
        log.write(json.dumps({"batch": bi, "harness": H, "branches": branches, "candidates": final}) + "\n")
        log.flush()
    log.close()

    # BARE baseline = stock mini-swe-agent (the paper's SWE baseline). READ-ONLY: measured once, offline,
    # and saved in logs/bare_cache.json; NEVER recomputed here (recomputing would burn API on a number we
    # already have). An instance absent from the cache simply does not count.
    bare_cache_path = PKG_DIR / "logs" / "bare_cache.json"
    bare_cache = json.load(open(bare_cache_path)) if bare_cache_path.exists() else {}
    bare_correct = sum(bare_cache.get(inst["instance_id"], False) for inst in items)

    print(f"\n######### RESULT (test-time / transductive) — final H = {H} #########", flush=True)
    print(f"  bare baseline (mini-swe-agent, from saved cache) = {bare_correct}/{len(items)}")
    print(f"  test-time evolved = {tt_correct}/{tt_total}")
    json.dump({"bare": bare_correct, "tt_correct": tt_correct, "tt_total": tt_total, "final_harness": H,
               "batches": tt_log, "per_problem": ev_results,
               "bare_by_id": {inst["instance_id"]: bare_cache.get(inst["instance_id"], False) for inst in items}},
              open(run_dir / "result.json", "w"), indent=2)
    print(f"[saved] {run_dir}/result.json   [traces] {run_dir}/traces/")


if __name__ == "__main__":
    main()
