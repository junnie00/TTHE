"""Run the BARE harness on a slice of DS-1000 problem_ids, GOLD-score, print accuracy, seed
logs/bare_cache.json (so ds1000_optimize reuses it).

Run (from the repo root):  OPENAI_API_KEY=... PYTHONPATH=. python -m ds1000.ds1000_bare <ids.json> <run_name>
"""
import json
import sys
from concurrent.futures import ThreadPoolExecutor

from . import ds1000_bridge as bridge
from .ds1000_common import load_harness, PKG_DIR
from .agents.bare import SYS as BARE_SYS


def main():
    ids = json.load(open(sys.argv[1]))
    run_name = sys.argv[2] if len(sys.argv) > 2 else "barehard"
    mode = sys.argv[3] if len(sys.argv) > 3 else "off"   # "off" (seed) | "on" (thinking-on fair baseline)
    probs = bridge.load_problems(ids=ids)
    cache_path = PKG_DIR / "logs" / ("bare_on_cache.json" if mode == "on" else "bare_cache.json")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache = json.load(open(cache_path)) if cache_path.exists() else {}
    print(f"[bare] {len(probs)} problems, run_name={run_name}, thinking={mode}, reusing {sum(1 for p in probs if str(p.pid) in cache)}", flush=True)

    def one(p):
        if str(p.pid) in cache:
            return p, cache[str(p.pid)]
        try:
            if mode == "on":
                code = bridge.extract_code(bridge.solver_llm(p.prompt, system=BARE_SYS, thinking="high"))
            else:
                code = load_harness("bare", p).solve()
        except Exception:  # noqa: BLE001
            code = ""
        ok = bool(bridge.is_correct(code, p)) if code else False
        return p, ok
    with ThreadPoolExecutor(max_workers=8) as ex:
        pairs = list(ex.map(one, probs))

    res = sum(ok for _, ok in pairs)
    print(f"\n######### BARE DS-1000 {run_name} (thinking={mode}): {res}/{len(pairs)} = {100*res/len(pairs):.1f}% #########", flush=True)
    cache = json.load(open(cache_path)) if cache_path.exists() else {}
    for p, ok in pairs:
        cache[str(p.pid)] = ok
    json.dump(cache, open(cache_path, "w"), indent=1)
    print(f"[bare] cached -> {cache_path}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
