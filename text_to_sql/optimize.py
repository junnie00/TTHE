"""TEST-TIME harness optimization — THE CURRENT LOOP (agentic, trace-driven; batch generate->judge).

This is the live entry point. ONE GENERAL harness (arbitrary Python; harness_base.SQLHarness) starts from
`bare` and ACCUMULATES across the stream. Gold is used ONLY to (a) pick the stream and (b) DISPLAY
before/after correctness — never inside the loop. Adaptation reads EXECUTION TRACES, not gold (label-free).

Per batch (--batch-size questions; scored transductively/test-time with the harness chosen FOR that batch):
  1. OBSERVE  — run each candidate harness on every batch question; write its FULL trace (question + Hint +
                schema + every coder call + every SQL + result + a BACK-TRANSLATION of the final SQL).
  2. GENERATE — --max-rounds rounds; each round spawns G=--group agentic generators (Claude Code on the
                frozen flash, WITH tools). Each deep-reads ALL candidates' traces + their source + probes
                the DB, then writes one improved harness .py. New candidates accumulate.
  3. PICK     — one agentic JUDGE deep-reads all traces, VERIFIES by running its OWN DB probes (no eyeball,
                no consensus, Hint is ground truth), and writes the single best harness name -> new H.

Other modules are diagnostics/ablations, NOT this loop: online.py / online_gold.py / supervised.py /
fixability.py / tt.py / evolve.py. NOTE evolve.py & online.py are ALSO shared libraries imported here
(load_harness, PKG_DIR, acc, ...), so they are dependencies — do not delete them.

    cd meta-harness-ref && ANTHROPIC_BASE_URL=... ANTHROPIC_AUTH_TOKEN=... TTHO_PROPOSER_MODEL=deepseek-v4-flash \
      OPENAI_API_KEY=... python -m text_to_sql.optimize \
      --db card_games --random 12 --group 2 --proposer agentic
"""
import argparse
import hashlib
import json
import os
import random
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import bridge
from . import proposer as P
from .bt import recover
from .evolve import load_harness, PKG, PKG_DIR, AGENTS_DIR, MH_ROOT


def isolated_solve(run_dir, harness_name, db_id, question, timeout, tag):
    """Run one harness solve in a killable process and return its full outcome.

    Generated harness code is untrusted: it may loop in Python, trigger a slow
    model call, or raise. A thread timeout cannot stop that work, so each solve
    owns a process group that the parent can terminate at the wall-clock limit.
    """
    jobs = Path(run_dir) / "solve_jobs"
    jobs.mkdir(parents=True, exist_ok=True)
    task_path = jobs / f"{tag}.task.json"
    output_path = jobs / f"{tag}.result.json"
    output_path.unlink(missing_ok=True)
    task_path.write_text(json.dumps({
        "harness": harness_name,
        "db_id": db_id,
        "question": question,
        "output": str(output_path),
    }), encoding="utf-8")

    started = time.monotonic()
    proc = subprocess.Popen(
        [sys.executable, "-u", "-m", f"{PKG}.solve_worker_v2", "--task", str(task_path)],
        cwd=str(MH_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        _stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        _stdout, stderr = proc.communicate()
        return {
            "status": "timeout",
            "sql": "",
            "result": {
                "ok": False,
                "rows": [],
                "error": f"wall timeout after {timeout}s",
            },
            "steps": [],
            "error": f"wall timeout after {timeout}s; process group killed",
            "duration_seconds": round(time.monotonic() - started, 3),
            "stderr": (stderr or "")[-2000:],
        }

    if not output_path.exists():
        return {
            "status": "worker_error",
            "sql": "",
            "result": {
                "ok": False,
                "rows": [],
                "error": f"worker exit={proc.returncode}; no result file",
            },
            "steps": [],
            "error": f"worker exit={proc.returncode}; no result file",
            "duration_seconds": round(time.monotonic() - started, 3),
            "stderr": (stderr or "")[-2000:],
        }
    try:
        outcome = json.loads(output_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "status": "worker_error",
            "sql": "",
            "result": {
                "ok": False,
                "rows": [],
                "error": f"invalid worker result: {exc}",
            },
            "steps": [],
            "error": f"invalid worker result: {exc}",
            "duration_seconds": round(time.monotonic() - started, 3),
            "stderr": (stderr or "")[-2000:],
        }
    outcome["worker_exit_code"] = proc.returncode
    outcome["stderr"] = (stderr or "")[-2000:]
    return outcome


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="card_games")
    ap.add_argument("--shuffle", type=int, default=1)
    ap.add_argument("--random", type=int, default=0, help=">0: stream N RANDOM (mixed right+wrong) questions")
    ap.add_argument("--cross-set", default=None, help="CROSS-DB stream: json {cross:[[db_id,idx],...]}")
    ap.add_argument("--cap", type=int, default=5, help="else: use the cached initial-coder-wrong set")
    ap.add_argument("--group", type=int, default=2, help="G agentic generators per GENERATE round (each writes one improved harness)")
    ap.add_argument("--max-rounds", type=int, default=3, help="GENERATE rounds per batch (each round's generators deep-read all traces so far)")
    ap.add_argument("--batch-size", type=int, default=1, help="number of TEST QUESTIONS evolved CONCURRENTLY (each independently from bare)")
    ap.add_argument("--proposer", choices=["single", "agentic"], default="agentic")
    ap.add_argument("--propose-timeout", type=int, default=400)
    ap.add_argument("--solve-timeout", type=int, default=90, help="hard wall-clock cap per harness.solve (anti-hang)")
    ap.add_argument("--model", default=os.environ.get("TTHO_PROPOSER_MODEL", "deepseek-v4-flash"))
    ap.add_argument("--run-name", default="opt1")
    ap.add_argument("--start-batch", type=int, default=0,
                    help="resume at this zero-based batch index")
    ap.add_argument("--initial-harness", default="bare",
                    help="harness carried into --start-batch")
    ap.add_argument("--prefix-results", default=None,
                    help="JSON file containing measured batch records before --start-batch")
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--refresh-bare", action="store_true", help="recompute the bare baseline even if cached in logs/bare_cache.json")
    ap.add_argument(
        "--bare-result",
        default=None,
        help="model-matched fixed bare result.json to reuse instead of re-calling the solver",
    )
    args = ap.parse_args()

    if args.fresh:
        for f in AGENTS_DIR.glob("cand_*.py"):   # only clear generated candidates; keep seed harnesses
            f.unlink()

    dbcache = {}

    def getdb(d):                                                    # per-DB: (db, schema) cached
        if d not in dbcache:
            database = bridge.get_db(d)
            dbcache[d] = (database, load_harness("bare", database).schema)
        return dbcache[d]

    if args.cross_set:                                              # CROSS-DB stream
        spec = json.load(open(args.cross_set))["cross"]            # [[db_id, idx], ...]
        items = [(d, bridge.eval_questions(d)[i]) for d, i in spec]
        # A cross-set index is meaningful only relative to the dataset file that
        # produced it.  When an adjacent details ledger exists, fail fast instead
        # of silently applying Mini-Dev indices to the full BIRD dev set.
        cross_path = Path(args.cross_set)
        details_path = cross_path.with_name(f"{cross_path.stem}_details.json")
        if details_path.exists():
            expected = json.loads(details_path.read_text(encoding="utf-8"))
            if len(expected) != len(items):
                raise ValueError(
                    f"cross-set identity mismatch: {len(items)} indexed items but "
                    f"{len(expected)} detail records in {details_path}"
                )
            mismatches = []
            for pos, ((db_id, question), detail) in enumerate(zip(items, expected)):
                actual_text = question.question.partition("\nHint:")[0].strip()
                expected_text = detail["question"].strip()
                if db_id != detail["db_id"] or actual_text != expected_text:
                    mismatches.append((pos, db_id, actual_text, detail["db_id"], expected_text))
            if mismatches:
                pos, db_id, actual_text, expected_db, expected_text = mismatches[0]
                raise ValueError(
                    "cross-set dataset identity mismatch "
                    f"({len(mismatches)}/{len(items)} items; first at position {pos}): "
                    f"loaded [{db_id}] {actual_text!r}, expected "
                    f"[{expected_db}] {expected_text!r}. "
                    "Set BIRD_DEV_FILE to the dataset file used to build the cross-set."
                )
    else:                                                          # single-DB (existing behaviour)
        qs_all = bridge.eval_questions(args.db)
        if args.random > 0:
            order = list(range(len(qs_all)))
            random.Random(args.shuffle).shuffle(order)
            sel = order[:args.random]
        else:
            cache = PKG_DIR / "logs" / f"wrong_{args.db}_shuf{args.shuffle}_n{args.cap}.json"
            sel = json.load(open(cache))["wrong_idx"] if cache.exists() else list(range(args.cap))
        items = [(args.db, qs_all[i]) for i in sel]
    golds = [bridge.gold_result(getdb(d)[0], q.gold_sql) for d, q in items]   # MEASUREMENT/DISPLAY ONLY

    run_dir = PKG_DIR / "logs" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    log = open(run_dir / "opt_log.jsonl", "w")
    branch_log = open(run_dir / "branch_lineage.jsonl", "w")
    interface = (PKG_DIR / "harness_base.py").read_text()
    print(f"\n######### TEST-TIME harness optimization (streaming; proposer={args.proposer} reads official traces; "
          f"controller alone executes each child) #########")
    print(f"[stream] {len(items)} questions across DBs: {sorted(set(d for d, _ in items))}", flush=True)

    # BATCH HARNESS EVOLUTION: the selected H carries across batches. Within one batch, initialize G fixed
    # branches from H. At every round all proposers see the same active branch--trace pairs, but proposer gi
    # edits only branch gi. After R rounds the judge sees only the final G branches. No gold enters this loop.
    for d, _q in items:                                              # pre-warm db cache single-threaded
        getdb(d)

    rn = args.run_name                                             # run-name prefix -> candidate files never collide across runs

    def write_trace(trace_dir, name, j, db_id, question, schema, outcome):
        """Write ONE harness's FULL trace on ONE batch question (question + Hint + schema + every step + back-translation)."""
        sql = outcome.get("sql", "")
        res = outcome.get("result", {"ok": False, "rows": []})
        steps = outcome.get("steps", [])
        qtext, _, hint = question.partition("\nHint:")
        L = [f"# Trace — harness `{name}` — Q{j}  [db={db_id}]\n",
             f"## QUESTION\n{qtext.strip()}\n",
             f"## HINT (AUTHORITATIVE — the harness MUST follow it exactly, never override it)\n{hint.strip() or '(none)'}\n",
             f"## DATABASE SCHEMA\n{schema}\n",
             f"## RUN STATUS\n{outcome.get('status', 'unknown')}\n",
             f"## FAILURE DETAIL\n{outcome.get('error', '') or '(none)'}\n",
             "## WHAT THE HARNESS DID — full step-by-step (every coder call + every SQL it ran, in order):"]
        for i, st in enumerate(steps, 1):
            if st.get("step") == "coder_llm":
                L.append(f"\n### step {i} — called the coder (deepseek)\nPROMPT:\n{str(st.get('prompt'))[:1500]}\n"
                         f"CODER RESPONSE:\n{str(st.get('response'))[:1500]}")
            else:
                L.append(f"\n### step {i} — executed SQL\nSQL: {str(st.get('sql'))[:4000]}\n"
                         f"RESULT: ok={st.get('ok')} n_rows={st.get('n_rows')} rows={str(st.get('rows'))[:300]} "
                         f"error={st.get('error')}")
        L.append(f"\n## FINAL SQL\n{str(sql)[:4000]}")
        L.append(f"\n## FINAL RESULT\n{P.desc(res)}")
        try:
            bt_en = recover(schema, sql) if sql else "(no SQL produced)"
        except Exception:
            bt_en = "(back-translation unavailable)"
        L.append(f"\n## BACK-TRANSLATION — what the FINAL SQL LITERALLY does, in plain English. COMPARE this to "
                 f"the QUESTION + HINT above: if the English describes something different from what the question "
                 f"asks for, the SQL is WRONG.\n{bt_en}")
        (trace_dir / f"{name}__q{j}.md").write_text("\n".join(L), encoding="utf-8")

    def observe(name, batch, trace_dir):
        """Run harness `name` on EVERY batch question IN PARALLEL, writing each question's full trace to
        trace_dir/<name>__q<j>.md. Each solve is isolated in a killable process.
        Returns list of execution results in batch order (for prequential scoring)."""
        results = [None] * len(batch)

        def one(jdbq):
            j, (db_id, q) = jdbq
            _db, schema = getdb(db_id)
            outcome = isolated_solve(
                run_dir,
                name,
                db_id,
                q.question,
                args.solve_timeout,
                f"b{bi}_{name}_q{j}",
            )
            write_trace(trace_dir, name, j, db_id, q.question, schema, outcome)
            results[j] = outcome["result"]

        with ThreadPoolExecutor(max_workers=min(len(batch), 16)) as ex:
            list(ex.map(one, list(enumerate(batch))))
        return results

    H = args.initial_harness
    B = args.batch_size
    batches = [items[i:i + B] for i in range(0, len(items), B)]
    batch_golds = [golds[i:i + B] for i in range(0, len(golds), B)]
    if not 0 <= args.start_batch < len(batches):
        raise ValueError(
            f"--start-batch must be in [0, {len(batches) - 1}], got {args.start_batch}"
        )
    if not _loadable(H, getdb(batches[args.start_batch][0][0])[0]):
        raise ValueError(f"--initial-harness is not loadable: {H}")

    prefix_log = []
    if args.prefix_results:
        prefix_data = json.loads(Path(args.prefix_results).read_text(encoding="utf-8"))
        prefix_log = prefix_data["batches"] if isinstance(prefix_data, dict) else prefix_data
    if args.start_batch == 0 and prefix_log:
        raise ValueError("--prefix-results must be empty when --start-batch=0")
    if args.start_batch > 0:
        if len(prefix_log) != args.start_batch:
            raise ValueError(
                f"--prefix-results must contain exactly {args.start_batch} batch records, "
                f"got {len(prefix_log)}"
            )
        for expected_batch, entry in enumerate(prefix_log):
            required = {"batch", "harness", "correct", "total"}
            if not required.issubset(entry):
                raise ValueError(
                    f"prefix batch {expected_batch} is missing {sorted(required - set(entry))}"
                )
            if entry["batch"] != expected_batch:
                raise ValueError(
                    f"prefix batches must be contiguous: expected {expected_batch}, "
                    f"got {entry['batch']}"
                )
            if entry["total"] != len(batches[expected_batch]):
                raise ValueError(
                    f"prefix batch {expected_batch} total={entry['total']} does not match "
                    f"dataset batch size {len(batches[expected_batch])}"
                )
        if prefix_log[-1]["harness"] != H:
            raise ValueError(
                "the last prefix harness must equal --initial-harness: "
                f"{prefix_log[-1]['harness']} != {H}"
            )

    config = {
        "cross_set": args.cross_set,
        "batch_size": args.batch_size,
        "group": args.group,
        "max_rounds": args.max_rounds,
        "proposer": args.proposer,
        "propose_timeout": args.propose_timeout,
        "solve_timeout": args.solve_timeout,
        "model": args.model,
        "run_name": args.run_name,
        "start_batch": args.start_batch,
        "initial_harness": H,
        "prefix_results": args.prefix_results,
        "batch_proposer_may_probe_db": True,
        "batch_proposer_may_execute_harness": False,
        "controller_executions_per_child_question": 1,
    }
    (run_dir / "config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )
    print(f"[batch-evolution] {len(batches)} batches of up to {B}; selected H accumulates across batches "
          f"(resume at batch {args.start_batch}, H={H}). Per batch: G={args.group} fixed branches, "
          f"R={args.max_rounds} branch-update rounds, "
          f"then 1 JUDGE picks only among the final branches. Full traces kept under {run_dir}/traces/.", flush=True)
    # TEST-TIME (transductive) accuracy: each batch ADAPTS the harness to its OWN label-free execution
    # evidence, then is scored with the harness CHOSEN FOR THAT batch, measured ON that same batch. Gold is
    # used ONLY for this measurement, never inside the loop (adaptation reads execution traces, not gold).
    # This is the standard test-time-adaptation protocol.
    tt_correct = sum(entry["correct"] for entry in prefix_log)
    tt_total = sum(entry["total"] for entry in prefix_log)
    tt_log = [dict(entry) for entry in prefix_log]
    for entry in prefix_log:
        log.write(json.dumps({
            **entry,
            "branches": [],
            "candidates": [entry["harness"]],
            "resumed_prefix": True,
        }) + "\n")
    log.flush()

    for bi in range(args.start_batch, len(batches)):
        batch = batches[bi]
        db0 = getdb(batch[0][0])[0]
        trace_dir = run_dir / "traces" / f"b{bi}"                  # ONE folder per batch (kept forever); each harness traced ONCE
        trace_dir.mkdir(parents=True, exist_ok=True)
        traced = set()
        cand_results = {}                                          # name -> exec results on THIS batch (traced once, cached)
        branches = [H] * args.group
        print(f"\n===== BATCH {bi}/{len(batches)} ({len(batch)} q) — start from H={H} =====", flush=True)
        # GENERATE phase: all branches share peer evidence, but each proposer receives its own branch as the
        # mechanically pre-copied edit base. A failed child leaves only that branch at its previous parent.
        for rnd in range(args.max_rounds):
            active = list(dict.fromkeys(branches))
            for c in active:
                if c not in traced and _loadable(c, db0):
                    cand_results[c] = observe(c, batch, trace_dir)
                    traced.add(c)
            proposed = P.sample_branches(
                branches,
                trace_dir,
                run_dir,
                f"b{bi}r{rnd}",
                rn,
                batch,
                args.model,
                args.propose_timeout,
                args.solve_timeout,
            )
            next_branches = []
            lineage = []
            for gi, (base, child) in enumerate(zip(branches, proposed)):
                accepted = child if child and _loadable(child, db0) else base
                next_branches.append(accepted)
                lineage.append({
                    "branch": gi,
                    "base_harness": base,
                    "base_trace_glob": str(trace_dir / f"{base}__q*.md"),
                    "proposed_child": child,
                    "active_child": accepted,
                    "child_trace_glob": str(trace_dir / f"{accepted}__q*.md"),
                    "proposal_card": (
                        str(P.proposal_card_path(run_dir, child))
                        if child else None
                    ),
                    "fell_back_to_base": accepted == base,
                })
            branches = next_branches
            for c in dict.fromkeys(branches):
                if c not in traced and _loadable(c, db0):
                    cand_results[c] = observe(c, batch, trace_dir)
                    traced.add(c)
            branch_log.write(json.dumps({
                "batch": bi,
                "round": rnd,
                "branches": lineage,
            }) + "\n")
            branch_log.flush()
            print(
                f"   batch{bi} gen-round{rnd}: "
                f"{sum(not row['fell_back_to_base'] for row in lineage)}/{args.group} branches advanced",
                flush=True,
            )

        # PICK phase: the judge sees only the final active branches, never the historical archive.
        final_candidates = list(dict.fromkeys(branches))
        for c in final_candidates:
            if c not in traced and _loadable(c, db0):
                cand_results[c] = observe(c, batch, trace_dir)
                traced.add(c)
        picked = P.pick_batch(
            final_candidates,
            trace_dir,
            run_dir,
            f"b{bi}",
            args.model,
            args.propose_timeout,
        )
        H = picked if picked in final_candidates else final_candidates[0]
        print(
            f"   batch{bi}: final branches={branches} "
            f"({len(final_candidates)} unique) -> JUDGE picked H={H}",
            flush=True,
        )
        # SCORE this batch with the harness CHOSEN FOR it (transductive / test-time) — result already traced.
        res = cand_results.get(H)
        if res is not None:
            batch_correct = sum(bridge.is_correct(r, g) for r, g in zip(res, batch_golds[bi]))
            tt_correct += batch_correct
            tt_total += len(batch)
            tt_log.append({"batch": bi, "harness": H, "correct": batch_correct, "total": len(batch)})
            print(f"   [test-time] batch{bi} H={H}: {batch_correct}/{len(batch)}", flush=True)
        log.write(json.dumps({
            "batch": bi,
            "harness": H,
            "branches": branches,
            "candidates": final_candidates,
        }) + "\n")
        log.flush()
    log.close()
    branch_log.close()
    # BARE BASELINE (no evolution), single-shot — the reference for "did adapting help".
    # Cache identity includes the dataset, solver model/endpoint, and bare harness digest;
    # question-only keys would silently reuse another model's baseline.
    bare_cache_path = PKG_DIR / "logs" / "bare_cache.json"
    bare_cache = json.load(open(bare_cache_path)) if (bare_cache_path.exists() and not args.refresh_bare) else {}
    bare_harness_digest = hashlib.sha1(
        (AGENTS_DIR / "bare.py").read_bytes()
    ).hexdigest()[:12]
    bare_identity = json.dumps(
        {
            "dataset_file": os.environ.get("BIRD_DEV_FILE", "dev.json"),
            "solver_model": os.environ.get("SOLVER_MODEL", "default"),
            "solver_base_url": os.environ.get("SOLVER_BASE_URL", "default"),
            "harness_digest": bare_harness_digest,
        },
        sort_keys=True,
    )

    def bkey(d, q):
        return hashlib.sha1(
            f"{bare_identity}\n{d}\n{q.question}".encode()
        ).hexdigest()

    if args.bare_result:
        fixed = json.loads(Path(args.bare_result).read_text(encoding="utf-8"))
        records = fixed.get("per_problem") or []
        if fixed.get("harness") != "bare" or len(records) != len(items):
            raise ValueError(
                "--bare-result must be a fixed bare result over exactly the current items"
            )
        for position, ((db_id, question), record) in enumerate(zip(items, records)):
            if record.get("position") != position or record.get("db_id") != db_id:
                raise ValueError(
                    f"--bare-result item mismatch at position {position}: {record}"
                )
            bare_cache[bkey(db_id, question)] = bool(record["correct"])
        bare_cache_path.write_text(json.dumps(bare_cache), encoding="utf-8")

    todo = [(d, q, g) for (d, q), g in zip(items, golds) if bkey(d, q) not in bare_cache]
    if todo:
        print(f"\n[baseline] running bare on {len(todo)} NEW question(s) (reusing {len(items) - len(todo)} cached) ...", flush=True)
        def bare_hit(arg):
            d, q, g = arg
            key = bkey(d, q)
            outcome = isolated_solve(
                run_dir,
                "bare",
                d,
                q.question,
                args.solve_timeout,
                f"baseline_bare_{key}",
            )
            r = outcome["result"]
            return bkey(d, q), bool(bridge.is_correct(r, g))
        with ThreadPoolExecutor(max_workers=16) as ex:
            for k, ok in ex.map(bare_hit, todo):
                bare_cache[k] = ok
        json.dump(bare_cache, open(bare_cache_path, "w"))
    else:
        print(f"\n[baseline] all {len(items)} questions cached — bare NOT re-run.", flush=True)
    bare_correct = sum(bare_cache[bkey(d, q)] for (d, q), _g in zip(items, golds))
    print(f"\n######### RESULT (test-time / transductive) — final harness H = {H} #########", flush=True)
    print(f"  bare baseline     = {bare_correct}/{len(items)}")
    print(f"  test-time evolved = {tt_correct}/{tt_total}  (each batch scored with the harness CHOSEN FOR it)")
    for entry in tt_log:
        print(f"    batch{entry['batch']} [{entry['harness']}]: {entry['correct']}/{entry['total']}")
    json.dump({"bare": bare_correct, "tt_correct": tt_correct, "tt_total": tt_total,
               "final_harness": H, "batches": tt_log,
               "resume": {
                   "start_batch": args.start_batch,
                   "initial_harness": args.initial_harness,
                   "prefix_results": args.prefix_results,
               }},
              open(run_dir / "result.json", "w"), indent=2)
    print(f"[saved] {run_dir}/result.json")


def _loadable(name, db):
    try:
        load_harness(name, db)
        return True
    except Exception:
        return False


if __name__ == "__main__":
    main()
