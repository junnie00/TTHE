"""Run the BARE harness (stock mini-swe-agent / flash) on a slice of instance-ids, score with the official
gold harness, print the resolved rate, and seed logs/bare_cache.json (so swe_optimize reuses it).

Run (from the monorepo root):  OPENAI_API_KEY=... PYTHONPATH=. python -m swebench.swe_bare <ids.json> <run_name>
"""
import json
import sys
from concurrent.futures import ThreadPoolExecutor

from . import swe_bridge as bridge
from .swe_common import load_harness, PKG_DIR


def main():
    ids = json.load(open(sys.argv[1]))
    run_name = sys.argv[2] if len(sys.argv) > 2 else "barehard"
    insts = bridge.load_instances(ids=ids)
    print(f"[bare] {len(insts)} instances, run_name={run_name}", flush=True)

    def one(inst):
        try:
            patch = load_harness("bare", inst).solve()
        except Exception:  # noqa: BLE001
            patch = ""
        return inst, (patch or "")
    with ThreadPoolExecutor(max_workers=8) as ex:
        pairs = list(ex.map(one, insts))
    print(f"[bare] rollouts done; non-empty patches = {sum(1 for _, p in pairs if p)}/{len(pairs)}", flush=True)

    gold = bridge.is_correct_batch(pairs, run_id=run_name)
    res = sum(bool(gold.get(i["instance_id"], False)) for i, _ in pairs)
    print(f"\n######### BARE {run_name}: {res}/{len(pairs)} resolved ({100*res/len(pairs):.1f}%) #########", flush=True)

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
