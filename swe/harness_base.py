"""The HARNESS interface for SWE-bench Verified (analogue of livecodebench/harness_base.py).

A harness is ARBITRARY PYTHON. The ONLY thing frozen is the LLM (model / endpoint / weights). Everything
else — the bash-interaction LOOP itself, the tool use, the prompts, the budget, and any post-rollout
verify->repair — is the harness's to evolve. `solve()` returns a git-diff PATCH. LABEL-FREE: the harness may
run anything in the REAL repo (self.exec: reproduction scripts, the repo's existing tests, git apply) and use
the execution behaviour as signal — but never the gold FAIL_TO_PASS/PASS_TO_PASS verdict (that is the answer
key, used outside for scoring only).
"""
from abc import ABC, abstractmethod

from . import swe_bridge as bridge


class SWEHarness(ABC):
    """Subclass this. You may rewrite ANY part of a harness — INCLUDING the whole agent loop: drive the
    frozen model with self.llm and run bash in the real repo with self.exec, or keep the stock loop via
    self.run_agent, or wrap it with a post-rollout verify->repair. Your action space is the whole Python
    file. Two invariants are fixed (audit_harness.py checks them; a candidate that breaks either is invalid):

      * FROZEN SOLVER — the LLM is fixed: never construct a new client/model, never change model / endpoint
        (base_url / api_key), never import a network library. You MAY freely change HOW you call it and the
        ENTIRE loop around it (prompts, number of llm/exec steps, tools, budget, voting, repair).
      * LABEL-FREE — the line is the ANSWER KEY, not the repo. You MAY run anything in the real repo via
        self.exec (apply a patch, run the agent's own reproduction script, run the repo's EXISTING test
        suite for regression) and use crash / test-error / diff behaviour as signal. You must NEVER use the
        gold verdict: no bridge.is_correct, no running the FAIL_TO_PASS/PASS_TO_PASS suite for pass/fail, no
        reading the reference patch or resolved status.

    Convenience helpers (use or ignore):
        self.instance       — the dataset dict (instance_id, repo, problem_statement, ...)
        self.problem        — the issue / PR description text
        self.repo           — the repo (e.g. 'pallets/flask')
        self.llm(messages) -> str
                            — ONE call to the FROZEN solver. messages=[{role, content}, ...].
        self.exec(command, timeout=None) -> {output, returncode}
                            — run one bash command in the REAL repo container (label-free). Build your OWN
                              read->exec->observe loop, or verify a patch (git apply; python -m pytest <the
                              repo's own tests>) after a rollout.
        self.run_agent(system_template=None, instance_template=None, step_limit=250, wall_time=5400) -> patch
                            — CONVENIENCE: one stock mini-swe-agent bash loop (the old default). You may keep
                              it, replace it with your own llm+exec loop, or wrap it with verify->repair.
        self._trace         — FULL trace of every llm / exec / rollout step.
    solve() must return the final patch (git-diff) string.
    """

    def __init__(self, instance):
        self.instance = instance
        self.problem = instance["problem_statement"]
        self.repo = instance["repo"]
        self._trace = []
        self._env = None          # real-repo Docker container (lazy; shared across llm+exec+run_agent)
        self._model = None        # frozen solver (lazy)

    def _get_env(self):
        if self._env is None:
            self._env = bridge.make_env(self.instance)
        return self._env

    def _get_model(self):
        if self._model is None:
            self._model = bridge.make_model()
        return self._model

    def llm(self, messages) -> str:
        """One call to the FROZEN solver. messages = [{'role': ..., 'content': ...}, ...]. Returns text."""
        out = bridge.model_query(self._get_model(), messages)
        self._trace.append({"step": "llm", "n_messages": len(messages), "response": str(out)[:1500]})
        return out

    def exec(self, command, timeout=None):
        """Run one bash command in the REAL repo container. Returns {output, returncode}. LABEL-FREE — the
        real repo is fair game (reproduction, regression on the repo's OWN tests, git apply); the gold
        FAIL_TO_PASS/PASS_TO_PASS verdict is not."""
        r = bridge.exec_in_repo(self._get_env(), command, timeout=timeout)
        self._trace.append({"step": "exec", "command": str(command)[:300],
                            "returncode": r["returncode"], "output": str(r["output"])[:800]})
        return r

    def run_agent(self, system_template=None, instance_template=None, step_limit=250, wall_time=5400) -> str:
        """CONVENIENCE: one stock mini-swe-agent rollout on the harness's shared env+model. Records the full
        trajectory into the trace. You may keep it, wrap it, or replace it with your own self.llm/self.exec
        loop — the loop is yours to evolve; only the model is frozen."""
        patch, info = bridge.run_stock_agent(self._get_env(), self._get_model(), self.instance,
                                             system_template, instance_template, step_limit, wall_time)
        self._trace.append({"step": "agent_rollout", "system": system_template, "instance_tmpl": instance_template,
                            "exit_status": info["exit_status"], "error": info.get("error", ""),
                            "n_calls": info["n_calls"], "messages": info["messages"], "patch": patch})
        return patch

    def cleanup(self):
        """Tear the container down. The optimizer calls this after solve(); safe to call twice."""
        if self._env is not None:
            bridge.teardown_env(self._env)
            self._env = None

    @abstractmethod
    def solve(self) -> str:
        """Return a git-diff PATCH (str) that fixes the issue."""
        ...
