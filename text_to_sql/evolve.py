"""Outer loop: TEST-TIME, LABEL-FREE Text-to-SQL harness evolution on the meta-harness framework.

Each iteration the proposer (Claude Code, driven by the meta-harness-sql skill) WRITES new harness .py
files. We score each on a LABEL-FREE fitness (metamorphic consistency) over an unlabeled BIRD stream and
keep a frontier. Gold accuracy is measured ALONGSIDE for the curve but NEVER enters selection or the
proposer's view. Research question: does selecting harnesses by the label-free signal also raise gold
accuracy (a rising curve), with a frozen weak solver and no training?

    cd meta-harness-ref && \
    ANTHROPIC_BASE_URL=https://api.deepseek.com/anthropic ANTHROPIC_AUTH_TOKEN=<dskey> \
    OPENAI_API_KEY=<k> \
    python -m tthe.evolve --db california_schools --shuffle 1 --cap 24 --iterations 5
"""
import argparse
import importlib
import json
import os
import random
import time
from pathlib import Path

from . import bridge
from . import claude_wrapper
from .evaluator import evaluate

PKG = "text_to_sql"
PKG_DIR = Path(__file__).parent
AGENTS_DIR = PKG_DIR / "agents"
MH_ROOT = PKG_DIR.parent   # repo root (parent of tthe/)
SKILL = PKG_DIR / ".claude/skills/meta-harness-sql"
BASELINE_AGENTS = {"__init__.py", "bare.py"}
PROPOSER_TOOLS = ["Read", "Glob", "Grep", "Write", "Edit", "Bash"]


def load_harness(name, db):
    """Import agents/<name>.py, instantiate its SQLHarness subclass with db."""
    from .harness_base import SQLHarness
    mod = importlib.import_module(f"{PKG}.agents.{name}")
    importlib.reload(mod)
    for obj in vars(mod).values():
        if (isinstance(obj, type) and issubclass(obj, SQLHarness) and obj is not SQLHarness
                and obj.__module__ == mod.__name__):
            return obj(db)
    raise ValueError(f"no SQLHarness subclass in agents/{name}.py")


def _exec_preview(harness, sql, n=3):
    r = bridge.execute(harness.db, sql)
    if not r["ok"]:
        return "(SQL ERROR)"
    if not r["rows"]:
        return "(empty)"
    return ("; ".join(str(x) for x in r["rows"][:n]) + (f" ...({len(r['rows'])} rows)" if len(r["rows"]) > n else ""))[:200]


def write_report(path, best_name, ev, k=6):
    """LABEL-FREE failure report: the best harness's lowest-fitness questions (NO gold)."""
    recs = sorted(ev["records"], key=lambda r: r["meta"])[:k]
    lines = [f"# Report: harness `{best_name}`  fitness={ev['fitness']:.3f} (inv={ev['inv']:.2f} sens={ev['sens']:.2f})",
             "",
             "Lowest label-free-fitness questions (your targets — raise INV/SENS / fix exec errors). "
             "INV=stable under rewording, SENS=changes under a counterfactual edit. NO gold is available.",
             ""]
    for i, r in enumerate(recs, 1):
        lines.append(f"## {i}. inv={r['inv']:.2f} sens={r['sens']:.2f}  meta={r['meta']:.2f}")
        lines.append(f"Question: {r['q']}")
        lines.append(f"Harness SQL: {r['sql'].strip()[:280]}")
        lines.append(f"Executed: {r['key'] if r['key'] is not None else '(ERROR/empty)'}")
        lines.append("")
    Path(path).write_text("\n".join(lines))


def task_prompt(iteration, run_dir, n):
    return (
        f"Run iteration {iteration} of LABEL-FREE Text-to-SQL harness evolution. Stream = {n} unlabeled "
        f"questions; you NEVER see gold.\n\n"
        f"## Run directory: {run_dir}\n"
        f"- `{run_dir}/evolution_summary.jsonl` — past candidates' label-free fitness\n"
        f"- `{run_dir}/frontier.json` — current best harness\n"
        f"- `{run_dir}/report.md` — the best harness's lowest-fitness questions (your targets)\n"
        f"- harness code lives in `tthe/agents/` (read the best ones; write new ones there)\n"
        f"- `tthe/harness_base.py` — the SQLHarness interface\n\n"
        f"Write your candidate list to `{run_dir}/pending_eval.json`. "
        f"To import-test a harness, run from `{MH_ROOT}`: "
        f"`PYTHONPATH={MH_ROOT} python -c \"import importlib; importlib.import_module('{PKG}.agents.NAME')\"`.\n"
        f"Follow the meta-harness-sql skill exactly. Implement 3 new harnesses."
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="california_schools")
    ap.add_argument("--shuffle", type=int, default=1)
    ap.add_argument("--cap", type=int, default=24)
    ap.add_argument("--iterations", type=int, default=5)
    ap.add_argument("--proposer-model", default=os.environ.get("TTHO_PROPOSER_MODEL", "deepseek-v4-flash"))
    ap.add_argument("--propose-timeout", type=int, default=1200)
    ap.add_argument("--npara", type=int, default=2)
    ap.add_argument("--ncf", type=int, default=2)
    ap.add_argument("--run-name", default="run1")
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()

    if args.fresh:
        for f in AGENTS_DIR.glob("*.py"):
            if f.name not in BASELINE_AGENTS:
                f.unlink()

    db = bridge.get_db(args.db)
    qs_all = bridge.eval_questions(args.db)
    order = list(range(len(qs_all)))
    random.Random(args.shuffle).shuffle(order)
    order = order[:args.cap]
    stream = [qs_all[i] for i in order]
    golds = [bridge.gold_result(db, q.gold_sql) for q in stream]   # MEASUREMENT ONLY

    run_dir = PKG_DIR / "logs" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    summary = run_dir / "evolution_summary.jsonl"
    frontier_f = run_dir / "frontier.json"
    report_f = run_dir / "report.md"
    if args.fresh:
        for f in (summary, frontier_f, report_f):
            f.unlink(missing_ok=True)

    def score(name):
        h = load_harness(name, db)
        ev = evaluate(h, stream, golds=golds, npara=args.npara, ncf=args.ncf)
        return h, ev

    print(f"\n########## TEST-TIME LABEL-FREE harness evolution (meta-harness) ##########")
    print(f"[db] {args.db} shuffle={args.shuffle} stream={len(stream)} iters={args.iterations} "
          f"proposer={args.proposer_model}")

    # seed: bare
    _, ev = score("bare")
    best = {"name": "bare", "fitness": ev["fitness"], "gold": ev["gold_acc"], "iteration": 0}
    frontier_f.write_text(json.dumps(best, indent=2))
    write_report(report_f, "bare", ev)
    with open(summary, "a") as f:
        f.write(json.dumps({"iteration": 0, "system": "bare", "fitness": round(ev["fitness"], 4),
                            "inv": round(ev["inv"], 3), "sens": round(ev["sens"], 3),
                            "gold_acc": round(ev["gold_acc"], 4)}) + "\n")
    print(f"[seed bare] fitness={ev['fitness']:.3f}  gold={ev['gold_acc']:.3f}")
    curve = [{"iteration": 0, "best": "bare", "fitness": ev["fitness"], "gold": ev["gold_acc"]}]

    env = os.environ.copy()
    env.pop("ANTHROPIC_API_KEY", None)   # use the deepseek Anthropic endpoint via BASE_URL/AUTH_TOKEN

    for it in range(1, args.iterations + 1):
        print(f"\n{'='*60}\n[iter {it}] proposer writing harnesses (best so far: {best['name']} "
              f"fit={best['fitness']:.3f} gold={best['gold']:.3f}) ...", flush=True)
        pending = run_dir / "pending_eval.json"
        pending.unlink(missing_ok=True)
        os.environ.update({k: env[k] for k in ("ANTHROPIC_BASE_URL", "ANTHROPIC_AUTH_TOKEN") if k in env})
        t0 = time.time()
        res = claude_wrapper.run(
            prompt=task_prompt(it, run_dir, len(stream)),
            model=args.proposer_model,
            allowed_tools=PROPOSER_TOOLS,
            skills=[str(SKILL)],
            cwd=str(MH_ROOT),
            log_dir=str(run_dir / "claude_sessions"),
            name=f"iter{it}",
            timeout_seconds=args.propose_timeout,
            progress=True,
        )
        print(f"  proposer done in {time.time()-t0:.0f}s exit={res.exit_code} "
              f"wrote={list(res.files_written.keys())[:6]}")
        if not pending.exists():
            print("  no pending_eval.json — skip iteration"); continue
        cands = json.loads(pending.read_text()).get("candidates", [])
        print(f"  proposed {len(cands)}: {[c['name'] for c in cands]}")

        for c in cands:
            name = c["name"]
            try:
                h, ev = score(name)
            except Exception as e:   # noqa: BLE001
                print(f"    {name}: INVALID ({str(e)[:80]})")
                with open(summary, "a") as f:
                    f.write(json.dumps({"iteration": it, "system": name, "invalid": str(e)[:120]}) + "\n")
                continue
            improved = ev["fitness"] > best["fitness"] + 1e-9
            print(f"    {name}: fitness={ev['fitness']:.3f} (inv={ev['inv']:.2f} sens={ev['sens']:.2f})  "
                  f"gold={ev['gold_acc']:.3f}  {'<= NEW BEST' if improved else ''}")
            with open(summary, "a") as f:
                f.write(json.dumps({"iteration": it, "system": name, "fitness": round(ev["fitness"], 4),
                                    "inv": round(ev["inv"], 3), "sens": round(ev["sens"], 3),
                                    "gold_acc": round(ev["gold_acc"], 4),
                                    "hypothesis": c.get("hypothesis", "")[:160]}) + "\n")
            if improved:
                best = {"name": name, "fitness": ev["fitness"], "gold": ev["gold_acc"], "iteration": it}
                frontier_f.write_text(json.dumps(best, indent=2))
                write_report(report_f, name, ev)      # next iter targets the new best's weak spots
        curve.append({"iteration": it, "best": best["name"], "fitness": best["fitness"], "gold": best["gold"]})
        print(f"  [frontier] {best['name']}  fitness={best['fitness']:.3f}  gold={best['gold']:.3f}")

    print(f"\n########## CURVE (label-free fitness vs gold, frontier per iteration) ##########")
    for c in curve:
        print(f"  iter {c['iteration']}: best={c['best']:<22} fitness={c['fitness']:.3f}  gold={c['gold']:.3f}")
    g0, gf = curve[0]["gold"], curve[-1]["gold"]
    print(f"\n  gold {g0:.3f} -> {gf:.3f}  ({gf-g0:+.3f})   "
          f"[does selecting on the LABEL-FREE signal raise true accuracy?]")
    json.dump({"db": args.db, "shuffle": args.shuffle, "order": order, "curve": curve},
              open(run_dir / "curve.json", "w"), indent=2)
    print(f"[saved] {run_dir}/curve.json")


if __name__ == "__main__":
    main()
