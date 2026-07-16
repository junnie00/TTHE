"""Isolated one-question harness runner used by ``optimize_v2``.

The worker is deliberately a separate process. A generated harness can hang,
raise, or trigger a model/API failure without poisoning the optimizer process.
The parent owns wall-clock timeouts and retries; this worker only records the
exact outcome of one attempt.
"""

import argparse
import json
import time
import traceback
from pathlib import Path

from . import bridge
from .evolve import load_harness


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    args = ap.parse_args()

    task = json.loads(Path(args.task).read_text(encoding="utf-8"))
    output = Path(task["output"])
    started = time.monotonic()
    payload = {
        "status": "worker_error",
        "sql": "",
        "result": {"ok": False, "rows": [], "error": "worker did not finish"},
        "steps": [],
        "error": "",
    }

    try:
        db = bridge.get_db(task["db_id"])
        harness = load_harness(task["harness"], db)
        sql = harness.solve(task["question"]) or ""
        result = bridge.execute(db, sql) if sql else {
            "ok": False,
            "rows": [],
            "error": "harness returned no SQL",
        }
        if not sql:
            status = "empty_sql"
        elif not result.get("ok"):
            status = "exec_error"
        elif not result.get("rows"):
            status = "empty_result"
        else:
            status = "ok"
        payload.update({
            "status": status,
            "sql": sql,
            "result": result,
            "steps": getattr(harness, "_trace", []),
            "error": str(result.get("error") or ""),
        })
    except Exception as exc:
        payload.update({
            "status": "exception",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(limit=20),
        })
    finally:
        payload["duration_seconds"] = round(time.monotonic() - started, 3)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, ensure_ascii=False, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
