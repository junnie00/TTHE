"""The HARNESS interface for SWE-bench Verified (analogue of livecodebench/harness_base.py).

A harness is ARBITRARY PYTHON wrapping a FROZEN mini-swe-agent rollout (deepseek-flash driving bash inside
the instance's Docker container). The proposer evolves the agent SCAFFOLD: the system/instance PROMPTS, the
step/time limits, and an optional post-rollout VERIFY->REPAIR. `solve()` returns a git-diff PATCH. LABEL-FREE:
it may only use the issue text and the agent's own in-container execution (its reproduction scripts + test
runs, visible in the trajectory) — never the gold FAIL_TO_PASS/PASS_TO_PASS tests (those are used outside for
scoring).
"""
from abc import ABC, abstractmethod

from . import swe_bridge as bridge


class SWEHarness(ABC):
    """Subclass this. You may rewrite ANY part of a harness — the agent's prompts (system_template,
    instance_template), step/time budget, how many rollouts, post-processing of the patch, verification —
    and `import bridge` to override call-layer parameters. Your action space is the whole Python file, NOT
    just the helpers below. Two invariants are fixed (audit_harness.py checks them; a candidate that breaks
    either is invalid):

      * FROZEN SOLVER — never change WHO answers: no new client / model, no reassigning the solver model or
        endpoint (base_url / api_key), no importing a network library. You MAY freely change HOW the agent
        is driven (prompts, step_limit, wall_time, number of rollouts).
      * LABEL-FREE — read only label-free signals: the agent's own message TRAJECTORY (reproduction scripts,
        test runs as bash observations) + the final PATCH. Never read gold: the resolved/pass status,
        bridge.is_correct, the reference patch, or any grading answer.

    Convenience helpers (use or ignore):
        self.instance       — the dataset dict (instance_id, repo, problem_statement, ...)
        self.problem        — the issue / PR description text
        self.repo           — the repo (e.g. 'pallets/flask')
        self.run_agent(system_template=None, instance_template=None, step_limit=80, wall_time=1500) -> patch
                            — ONE frozen mini-swe-agent rollout in the Docker container; returns a git-diff
        self._trace         — FULL trace: every rollout (prompts, exit_status, n_calls, messages, patch)
    solve() must return the final patch (git-diff) string.
    """

    def __init__(self, instance):
        self.instance = instance
        self.problem = instance["problem_statement"]
        self.repo = instance["repo"]
        self._trace = []

    def run_agent(self, system_template=None, instance_template=None, step_limit=80, wall_time=1500) -> str:
        """Run ONE frozen mini-swe-agent rollout (custom prompts optional; default -> stock swebench.yaml).
        Records the full trajectory into the trace so the proposer can see/evolve it. Returns the patch."""
        patch, info = bridge.agent_rollout(self.instance, system_template, instance_template, step_limit, wall_time)
        self._trace.append({"step": "agent_rollout", "system": system_template, "instance_tmpl": instance_template,
                            "exit_status": info["exit_status"], "n_calls": info["n_calls"],
                            "messages": info["messages"], "patch": patch})
        return patch

    @abstractmethod
    def solve(self) -> str:
        """Return a git-diff PATCH (str) that fixes the issue."""
        ...
