"""Dataset interface. DemoDataset runs offline; BirdDataset loads BIRD dev.

Only `eval_questions()` exposes the real labeled questions, and it is used ONLY for the
final evaluation. The evolution loop never calls it (label-free, see §0.5.1).
"""
import os
import json
import random
from dataclasses import dataclass

from .db import Database
from .demo_db import build_demo_db


@dataclass
class LabeledQuestion:
    db_id: str
    question: str
    gold_sql: str


class Dataset:
    def list_db_ids(self):
        raise NotImplementedError

    def get_database(self, db_id) -> Database:
        raise NotImplementedError

    def eval_questions(self, db_id):
        """Real labeled questions — FINAL EVALUATION ONLY."""
        raise NotImplementedError


class DemoDataset(Dataset):
    def __init__(self, workdir):
        self.path = build_demo_db(os.path.join(workdir, "demo.sqlite"))

    def list_db_ids(self):
        return ["demo"]

    def get_database(self, db_id="demo"):
        return Database("demo", self.path)

    def eval_questions(self, db_id="demo"):
        return [
            LabeledQuestion("demo", "How many singers are there?",
                            "SELECT count(*) FROM singer"),
            LabeledQuestion("demo", "What are the names of singers who performed in a stadium with capacity over 50000?",
                            "SELECT DISTINCT T1.name FROM singer T1 "
                            "JOIN concert T2 ON T1.singer_id=T2.singer_id "
                            "JOIN stadium T3 ON T2.stadium_id=T3.stadium_id WHERE T3.capacity>50000"),
            LabeledQuestion("demo", "How many concerts were held in the stadium named 'Bird Nest'?",
                            "SELECT count(*) FROM concert T1 JOIN stadium T2 "
                            "ON T1.stadium_id=T2.stadium_id WHERE T2.name='Bird Nest'"),
        ]


class BirdDataset(Dataset):
    """BIRD dev loader.

    Expects, under `root`:
        dev.json                       (list of {db_id, question, SQL, ...})
        dev_databases/<db_id>/<db_id>.sqlite   (or database/<db_id>/<db_id>.sqlite)
    Adjust _db_dir / field names if your BIRD copy differs.
    """
    def __init__(self, root):
        self.root = root
        dev_name = os.environ.get("BIRD_DEV_FILE", "dev.json")   # set BIRD_DEV_FILE=mini_dev.json for Mini-Dev (500)
        dev_path = os.path.join(root, dev_name)
        if not os.path.exists(dev_path):
            raise FileNotFoundError(f"BIRD dev file not found at {dev_path} — set dataset.bird_root / BIRD_DEV_FILE")
        with open(dev_path, encoding="utf-8") as f:
            self.dev = json.load(f)

    def _db_dir(self):
        for cand in ("dev_databases", "database", "databases"):
            p = os.path.join(self.root, cand)
            if os.path.isdir(p):
                return p
        return os.path.join(self.root, "dev_databases")

    def list_db_ids(self):
        return sorted({e["db_id"] for e in self.dev})

    def get_database(self, db_id):
        path = os.path.join(self._db_dir(), db_id, f"{db_id}.sqlite")
        return Database(db_id, path)

    def eval_questions(self, db_id):
        out = []
        for e in self.dev:
            if e["db_id"] == db_id:
                q = e["question"]
                ev = (e.get("evidence") or "").strip()
                if ev:                       # BIRD evidence = external knowledge needed to answer
                    q = f"{q}\nHint: {ev}"
                out.append(LabeledQuestion(db_id, q, e.get("SQL") or e.get("query", "")))
        return out


def build_dataset(ds_cfg, workdir):
    if ds_cfg["name"] == "bird":
        return BirdDataset(ds_cfg["bird_root"])
    if ds_cfg["name"] == "demo":
        return DemoDataset(workdir)
    raise ValueError(f"unknown dataset.name={ds_cfg['name']}")


def sample_train_questions(train_path, k=5):
    """Sample real questions from a DISJOINT corpus (BIRD train: different dbs than dev) to give
    the examiner a real-question STYLE prior. We use only question text + evidence — never the SQL,
    and never the target db's questions — so this stays label-free for the target database.
    k=None returns the whole pool (used to load the hard-train pool the proposer samples from).
    """
    with open(train_path, encoding="utf-8") as f:
        data = json.load(f)
    chosen = data if (k is None or k >= len(data)) else random.sample(data, k)
    out = []
    for e in chosen:
        ev = (e.get("evidence") or "").strip()
        out.append(e["question"] + (f" (hint: {ev})" if ev else ""))
    return out
