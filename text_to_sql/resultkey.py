"""Canonical, hashable key for a SQL execution result.

Two harnesses AGREE iff their results share the same non-None key. Errors / empty results map to
None (a failure never counts as agreement). Matches the order-insensitive set-comparison semantics
used for gold scoring.
"""


def result_key(rows, ok, maxn=2000):
    """Huge results (> maxn rows, almost always a cartesian/wrong answer) get a cheap size-based key
    instead of sorting 100k strings; a BIG result won't match the small gold anyway, so it is still
    scored wrong."""
    if not ok or not rows:
        return None
    if len(rows) > maxn:
        return f"BIG:{len(rows)}"
    return str(sorted(str(x) for x in rows))[:4000]
