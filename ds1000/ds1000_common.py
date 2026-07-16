"""Shared constants + harness loader for the DS-1000 domain (analogue of livecodebench/lcb_common.py).
The proposer writes DS1000Harness subclasses into agents/<name>.py; this reloads and instantiates them."""
import importlib
from pathlib import Path

PKG = "ds1000"
PKG_DIR = Path(__file__).parent
AGENTS_DIR = PKG_DIR / "agents"
MH_ROOT = PKG_DIR.parent          # monorepo root (parent of this domain dir)

from .harness_base import DS1000Harness


def load_harness(name, problem):
    """Import + RELOAD agents/<name>.py, find its DS1000Harness subclass, instantiate it with `problem`."""
    mod = importlib.import_module(f"{PKG}.agents.{name}")
    importlib.reload(mod)
    for v in vars(mod).values():
        if isinstance(v, type) and issubclass(v, DS1000Harness) and v is not DS1000Harness:
            return v(problem)
    raise ValueError(f"no DS1000Harness subclass found in agents/{name}.py")
