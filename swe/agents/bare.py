"""Baseline harness: one stock mini-swe-agent rollout (default swebench.yaml prompts, no custom workflow,
no post-rollout repair). This is the floor every proposed harness is measured against — the official agent."""
from ..harness_base import SWEHarness
from .. import swe_bridge as bridge


class BareHarness(SWEHarness):
    def solve(self) -> str:
        return self.run_agent()        # all defaults -> bridge uses the stock swebench.yaml prompts
