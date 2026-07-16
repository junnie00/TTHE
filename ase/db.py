"""SQLite layer: schema introspection, safe SELECT execution, result comparison.

The only verifier in this project is SQL execution on the real database (label-free,
deterministic). Everything that needs a "ground truth" goes through here.
"""
import re
import sqlite3

_FORBIDDEN = re.compile(r"\b(drop|delete|update|insert|alter|create|replace|attach|detach|pragma|vacuum)\b", re.I)


class Database:
    def __init__(self, db_id, sqlite_path):
        self.db_id = db_id
        self.sqlite_path = sqlite_path
        self.schema = introspect(sqlite_path)

    def execute(self, sql, timeout=5.0, limit=2000):
        return execute_sql(self.sqlite_path, sql, timeout, limit)

    def schema_text(self):
        lines = []
        for t in self.schema["tables"]:
            cols = ", ".join(c["name"] for c in t["columns"])
            lines.append(f"Table {t['name']}({cols})")
        for fk in self.schema["foreign_keys"]:
            lines.append(f"FK {fk['from_table']}.{fk['from_col']} -> {fk['to_table']}.{fk['to_col']}")
        return "\n".join(lines)

    def join_graph_text(self):
        edges = sorted({f"{fk['from_table']}-{fk['to_table']}" for fk in self.schema["foreign_keys"]})
        return "; ".join(edges) or "(no foreign keys)"

    def text_columns(self):
        out = []
        for t in self.schema["tables"]:
            for c in t["columns"]:
                if "char" in c["type"].lower() or "text" in c["type"].lower() or c["type"] == "":
                    out.append((t["name"], c["name"]))
        return out

    def sample_values(self, table, col, k=3):
        r = self.execute(f'SELECT DISTINCT "{col}" FROM "{table}" WHERE "{col}" IS NOT NULL LIMIT {k}')
        return [row[0] for row in r["rows"]] if r["ok"] else []


def introspect(path):
    con = sqlite3.connect(path)
    con.text_factory = lambda b: b.decode(errors="ignore") if isinstance(b, bytes) else b
    cur = con.cursor()
    tables, fks = [], []
    names = [r[0] for r in cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]
    for t in names:
        cols = []
        for _cid, name, ctype, _nn, _df, pk in cur.execute(f'PRAGMA table_info("{t}")'):
            cols.append({"name": name, "type": ctype or "", "pk": bool(pk)})
        tables.append({"name": t, "columns": cols})
        for row in cur.execute(f'PRAGMA foreign_key_list("{t}")'):
            # row = (id, seq, ref_table, from_col, to_col, on_update, on_delete, match)
            fks.append({"from_table": t, "from_col": row[3], "to_table": row[2], "to_col": row[4]})
    con.close()
    return {"tables": tables, "foreign_keys": fks}


def execute_sql(path, sql, timeout=5.0, limit=2000):
    sql = (sql or "").strip().rstrip(";").strip()
    if not sql:
        return {"ok": False, "rows": [], "error": "empty sql", "n": 0}
    if ";" in sql or _FORBIDDEN.search(sql):
        return {"ok": False, "rows": [], "error": "only a single read-only SELECT is allowed", "n": 0}
    if not re.match(r"(?is)^\s*(select|with)\b", sql):
        return {"ok": False, "rows": [], "error": "not a SELECT", "n": 0}
    try:
        con = sqlite3.connect(path, timeout=timeout)
        con.text_factory = lambda b: b.decode(errors="ignore") if isinstance(b, bytes) else b
        cur = con.cursor()
        cur.execute(sql)
        rows = cur.fetchmany(limit)
        con.close()
        return {"ok": True, "rows": [tuple(r) for r in rows], "error": None, "n": len(rows)}
    except Exception as e:  # noqa: BLE001 - surface any SQL error as a string
        return {"ok": False, "rows": [], "error": str(e), "n": 0}


def compare_results(a, b):
    """Order-insensitive set-of-rows equality (column order within a row is preserved).

    NOTE: this is the simple execution-accuracy proxy. BIRD's official metric is similar
    (set comparison of result rows). Tighten here if you need exact BIRD parity.
    """
    return _norm(a) == _norm(b)


def _norm(rows):
    return {tuple(str(x) for x in row) for row in rows}
