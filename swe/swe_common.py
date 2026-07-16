"""Shared constants + harness loader for the SWE-bench Verified domain (analogue of livecodebench/lcb_common.py).
The proposer writes SWEHarness subclasses into agents/<name>.py; this reloads and instantiates them."""
import importlib
from pathlib import Path

PKG = "swe"
PKG_DIR = Path(__file__).parent
AGENTS_DIR = PKG_DIR / "agents"
# Monorepo root = parent of this domain's directory (…/TTHE_mono). Kept importable as sys.path root.
REPO_ROOT = PKG_DIR.parent
MH_ROOT = REPO_ROOT  # backward-compat alias used by swe_proposer

from .harness_base import SWEHarness


def load_harness(name, instance):
    """Import + RELOAD agents/<name>.py, find its SWEHarness subclass, instantiate it with `instance`."""
    mod = importlib.import_module(f"{PKG}.agents.{name}")
    importlib.reload(mod)
    for v in vars(mod).values():
        if isinstance(v, type) and issubclass(v, SWEHarness) and v is not SWEHarness:
            return v(instance)
    raise ValueError(f"no SWEHarness subclass found in agents/{name}.py")
