"""Run the BARE harness (stock mini-swe-agent / flash) on a slice of instance-ids, score with the official
gold harness, print the resolved rate, and seed logs/bare_cache.json (so swe_optimize reuses it).

Run (from the monorepo root):  OPENAI_API_KEY=... PYTHONPATH=. python -m swebench.swe_bare <ids.json> <run_name>
"""
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

from . import swe_bridge as bridge
from .swe_common import load_harness, PKG_DIR



def _render_msg(m):
    """Render ONE agent message in full.

    mini-swe-agent v2.4.2 drives the model with TOOL CALLS, so an assistant turn carries its action in
    `tool_calls` and its `content` is usually empty — reading only `content` dropped 74 of 80 assistant
    steps (92%) from the trace. What survived was a list of command outputs with no visible reasoning or
    commands, which is precisely what a proposer needs in order to diagnose a failure. Thinking models
    also route their text to `reasoning_content` with `content` null, so that is included too."""
    parts = []
    rc = m.get("reasoning_content")
    if rc:
        parts.append(f"[reasoning]\n{rc}")
    c = m.get("content")
    if isinstance(c, list):                       # block-style content
        c = "\n".join(str(b.get("text", b)) if isinstance(b, dict) else str(b) for b in c)
    if c:
        parts.append(str(c))
    for tc in (m.get("tool_calls") or []):
        fn = (tc or {}).get("function", {}) if isinstance(tc, dict) else {}
        parts.append(f"[tool_call] {fn.get('name', '?')}({fn.get('arguments', '')})")
    return "\n".join(parts) if parts else "(empty)"


def main():
    ids = json.load(open(sys.argv[1]))
    run_name = sys.argv[2] if len(sys.argv) > 2 else "barehard"
    insts = bridge.load_instances(ids=ids)
    print(f"[bare] {len(insts)} instances, run_name={run_name}", flush=True)

    # A baseline that persists only a verdict cannot be re-verified or diagnosed later, and every
    # "why did it fail" question then needs a fresh (expensive) Docker rollout. Keep the whole thing.
    run_dir = PKG_DIR / "logs" / f"bare_{run_name}"
    (run_dir / "traces").mkdir(parents=True, exist_ok=True)

    def write_trace(inst, patch, steps):
        iid = inst["instance_id"]
        roll = next((st for st in steps if st.get("step") == "agent_rollout"), {})
        # Surface the rollout's OUTCOME at the top. An empty patch caused by an exhausted API budget and
        # an empty patch caused by the model failing look identical without it.
        L = [f"# Baseline trace — harness `bare` — {iid}  [{inst['repo']}]\n",
             f"## ROLLOUT OUTCOME\n  exit_status = {roll.get('exit_status')!r}\n"
             f"  n_calls = {roll.get('n_calls')}\n  error = {str(roll.get('error', ''))[:800]!r}\n",
             f"## ISSUE\n{inst['problem_statement']}\n",
             "## WHAT THE AGENT DID — full trajectory:"]
        msgs = [m for st in steps if st.get("step") == "agent_rollout" for m in st.get("messages", [])
                if m.get("role") != "system"]
        for i, m in enumerate(msgs, 1):
            L.append(f"\n### step {i} ({m.get('role')})\n{_render_msg(m)}")
        L.append(f"\n## FINAL PATCH\n```diff\n{patch}\n```")
        (run_dir / "traces" / f"{iid}.md").write_text("\n".join(L), encoding="utf-8")

    def one(inst):
        h, patch = None, ""
        try:
            h = load_harness("bare", inst)
            patch = h.solve() or ""
        except Exception as e:  # noqa: BLE001
            print(f"  [bare] {inst['instance_id']} CRASHED: {type(e).__name__}: {e}", flush=True)
        finally:
            try:
                write_trace(inst, patch, getattr(h, "_trace", []) if h else [])
                if h:
                    h.cleanup()          # a leaked container per instance exhausts the box
            except Exception:  # noqa: BLE001
                pass
        print(f"  [bare] {inst['instance_id']}: patch {len(patch)} chars", flush=True)
        return inst, patch
    with ThreadPoolExecutor(max_workers=int(os.environ.get("SWE_WORKERS", "8"))) as ex:
        pairs = list(ex.map(one, insts))
    print(f"[bare] rollouts done; non-empty patches = {sum(1 for _, p in pairs if p)}/{len(pairs)}", flush=True)

    gold = bridge.is_correct_batch(pairs, run_id=run_name)
    res = sum(bool(gold.get(i["instance_id"], False)) for i, _ in pairs)
    print(f"\n######### BARE {run_name}: {res}/{len(pairs)} resolved ({100*res/len(pairs):.1f}%) #########", flush=True)
    json.dump({"kind": "bare_baseline", "run_name": run_name, "resolved": res, "total": len(pairs),
               "per_instance": [{"instance_id": i["instance_id"], "repo": i["repo"],
                                 "resolved": bool(gold.get(i["instance_id"], False)),
                                 "patch": p} for i, p in pairs]},
              open(run_dir / "result.json", "w"), indent=2)
    print(f"[bare] traces + result.json -> {run_dir}", flush=True)

    cache_path = PKG_DIR / "logs" / "bare_cache.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache = json.load(open(cache_path)) if cache_path.exists() else {}
    for i, _ in pairs:
        cache[i["instance_id"]] = bool(gold.get(i["instance_id"], False))
    json.dump(cache, open(cache_path, "w"), indent=1)
    print(f"[bare] cached -> {cache_path}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
