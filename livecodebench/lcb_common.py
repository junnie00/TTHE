"""Shared constants + harness loader for the LiveCodeBench domain (analogue of text_to_sql/evolve.py's
load_harness/PKG bits). The proposer writes CodeHarness subclasses into agents/<name>.py; this reloads
and instantiates them."""
import importlib
from pathlib import Path

PKG = "livecodebench"
PKG_DIR = Path(__file__).parent
AGENTS_DIR = PKG_DIR / "agents"
MH_ROOT = PKG_DIR.parent          # monorepo REPO_ROOT (parent of this domain dir)

from .harness_base import CodeHarness


def load_harness(name, problem):
    """Import + RELOAD agents/<name>.py, find its CodeHarness subclass, instantiate it with `problem`."""
    mod = importlib.import_module(f"{PKG}.agents.{name}")
    importlib.reload(mod)
    for v in vars(mod).values():
        if isinstance(v, type) and issubclass(v, CodeHarness) and v is not CodeHarness:
            return v(problem)
    raise ValueError(f"no CodeHarness subclass found in agents/{name}.py")
