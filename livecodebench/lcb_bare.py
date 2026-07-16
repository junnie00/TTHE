"""Compute the BARE baseline on a LiveCodeBench slice with PER-QID caching (provenance for the paper).
Mirrors swebench/swe_bare.py and ds1000/ds1000_bare.py.

  OPENAI_API_KEY=... PYTHONPATH=. python -m livecodebench.lcb_bare \
      livecodebench/logs/hard60.json <run_name> [on|off]

Reasoning-ON writes logs/bare_on_cache.json; reasoning-OFF writes logs/bare_off_cache.json (kept separate
from the loop's own logs/bare_cache.json). Cache keyed by str(qid); reused across calls.
"""
import json
import sys
from concurrent.futures import ThreadPoolExecutor

from . import lcb_bridge as bridge
from .lcb_common import PKG_DIR
from .agents.bare import SYS


def main():
    spec = json.load(open(sys.argv[1]))
    qids = [it["qid"] for it in spec["items"]] if isinstance(spec, dict) else [
        (it["qid"] if isinstance(it, dict) else it) for it in spec]
    run_name = sys.argv[2] if len(sys.argv) > 2 else "bare_hard"
    mode = sys.argv[3] if len(sys.argv) > 3 else "on"          # "on" | "off"
    thinking = "high" if mode == "on" else False

    allp = {p.qid: p for p in bridge.load_problems("test6", stdin_only=True)}
    probs = [allp[q] for q in qids]
    cache_path = PKG_DIR / "logs" / ("bare_on_cache.json" if mode == "on" else "bare_off_cache.json")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache = json.load(open(cache_path)) if cache_path.exists() else {}
    todo = [p for p in probs if str(p.qid) not in cache]
    print(f"[bare] {len(probs)} problems, thinking={mode}, reuse {len(probs)-len(todo)}, compute {len(todo)}", flush=True)

    def one(p):
        prompt = f"{p.content}\n\nWrite the complete Python 3 solution (read stdin, print stdout)."
        code = bridge.extract_code(bridge.solver_llm(prompt, system=SYS, thinking=thinking))
        ok = bool(bridge.is_correct(code, p)) if code else False
        return str(p.qid), ok
    if todo:
        with ThreadPoolExecutor(max_workers=6) as ex:
            for qid, ok in ex.map(one, todo):
                cache[qid] = ok
        json.dump(cache, open(cache_path, "w"), indent=1)
    res = sum(bool(cache.get(str(p.qid), False)) for p in probs)
    print(f"\n######### BARE LCB {run_name} (thinking={mode}): {res}/{len(probs)} = {100*res/len(probs):.1f}% #########", flush=True)
    print(f"[bare] cached -> {cache_path}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
