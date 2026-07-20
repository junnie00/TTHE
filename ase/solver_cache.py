"""Disk-backed cache of FROZEN-solver replies, shared by every harness in a TTHE run.

WHY THIS EXISTS — it removes a noise source large enough to swamp the effect the loop measures.

Measured on DS-1000 (ds_b0_fixed batch0): of 134 coder calls, only 27 prompts were DISTINCT; one identical
prompt was issued 20 times. Candidates that never touched the prompt-building code were therefore asked the
SAME question and handed DIFFERENT answers, because thinking mode is nondeterministic even at temperature 0.
`react`, `b0r0_g0` and `b0r0_g2` have byte-identical system prompts AND byte-identical user prompts on all
four discriminating problems of that batch, yet scored 6/10, 3/10 and 4/10. A +-3 spread from pure sampling,
on a 10-problem batch, is larger than any improvement a proposer can realistically produce.

With the cache: harnesses that ask the same thing get the same answer, so a score difference can ONLY come
from a real behavioural difference. Measured on the next run: 0 cases of one prompt yielding two answers,
55% fewer API calls.

A second, unplanned benefit — it makes NULL RESULTS VISIBLE. In ds_full50 batch3, all ten candidates emitted
byte-identical code on 7 of 10 problems: the round had produced no behavioural change at all, despite each
candidate editing 9-113 lines. Without the cache, ten behaviourally identical harnesses would have scored
differently by luck and the batch would have been misread as a difficulty effect.

USAGE — wrap the raw request, do not replace it:

    from ase.solver_cache import SolverCache
    _CACHE = SolverCache(os.environ.get("LCB_SOLVER_CACHE", ".../logs/solver_cache.json"))
    ...
    outs = [_CACHE.get_or_call((prompt, system, thinking, mt, seq, i), one) for i in range(n)]

THE `seq` FIELD IS LOAD-BEARING. It is how many times THIS harness instance has already issued THIS exact
request, and it is what stops the cache from destroying deliberate resampling: a harness that asks the same
question three times in order to vote gets seq=0,1,2 and therefore three DIFFERENT replies, while a second
harness doing the same thing gets the same three. Drop `seq` and all three collapse into one answer,
silently deleting the mechanism the harness was built around. The harness base class maintains the counter
(see DS1000Harness.llm).
"""
import hashlib
import json
import os
import threading


class SolverCache:
    """Thread-safe, process-shared, disk-backed. Safe to construct at import time; loads lazily."""

    def __init__(self, path):
        self.path = path
        self._lock = threading.Lock()
        self._data = None

    def _load(self):
        if self._data is None:
            with self._lock:
                if self._data is None:
                    try:
                        self._data = json.load(open(self.path, encoding="utf-8"))
                    except Exception:  # noqa: BLE001  — absent or corrupt cache is simply an empty one
                        self._data = {}
        return self._data

    @staticmethod
    def key(parts):
        """Stable key over an arbitrary tuple of request parameters. Everything that changes WHAT the model
        is asked, or WHICH sample of the answer is wanted, must be in here."""
        return hashlib.sha1(json.dumps([str(p) for p in parts], ensure_ascii=False).encode("utf-8")).hexdigest()

    def get_or_call(self, parts, produce):
        """Return the cached reply for `parts`, else call `produce()` and store it."""
        k = self.key(parts)
        data = self._load()
        if k in data:
            return data[k]
        value = produce()
        with self._lock:
            data[k] = value
            try:
                d = os.path.dirname(self.path)
                if d:
                    os.makedirs(d, exist_ok=True)
                tmp = self.path + ".tmp"
                with open(tmp, "w", encoding="utf-8") as fh:
                    json.dump(data, fh)
                os.replace(tmp, self.path)   # atomic: concurrent harnesses must never read a half-written file
            except Exception:  # noqa: BLE001  — a cache that cannot persist must not break the run
                pass
        return value

    def __len__(self):
        return len(self._load())
