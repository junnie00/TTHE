from __future__ import annotations

import json
import copy
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from audit_agent import audit_source  # noqa: E402
from tthe_bridge import build_harbor_command, collect_harbor_observations  # noqa: E402
from tthe_proposer import _judge_prompt, _prepare_judge_provenance  # noqa: E402


class BridgeTests(unittest.TestCase):
    def test_safe_docker_is_explicitly_selected_only_when_enabled(self):
        with patch.dict(os.environ, {"TTHE_SAFE_DOCKER": "1"}):
            command = build_harbor_command(
                "harbor", "agents.baseline_kira:AgentHarness", "job", Path("/tmp/jobs"),
                ["extract-elf"], dataset="terminal-bench@2.0", model="openai/deepseek-chat",
                environment="docker", concurrency=1, max_turns=100,
            )
        self.assertEqual(command[command.index("--environment-import-path") + 1],
                         "safe_docker:SafeDockerEnvironment")

    def test_harbor_command_is_one_attempt_and_task_scoped(self):
        command = build_harbor_command(
            "uv run harbor",
            "agents.baseline_kira:AgentHarness",
            "job",
            Path("/tmp/jobs"),
            ["extract-elf", "regex-chess"],
            dataset="terminal-bench@2.0",
            model="anthropic/claude-opus-4-6",
            environment="runloop",
            concurrency=2,
            max_turns=8,
        )
        self.assertEqual(command[:3], ["uv", "run", "harbor"])
        self.assertIn("--n-attempts", command)
        self.assertEqual(command[command.index("--n-attempts") + 1], "1")
        self.assertIn("--yes", command)
        self.assertEqual(
            command[command.index("--agent-setup-timeout-multiplier") + 1], "5"
        )
        kwargs = [
            command[index + 1]
            for index, value in enumerate(command)
            if value == "--agent-kwarg"
        ]
        self.assertEqual(kwargs, ["record_terminal_session=false", "max_turns=8"])
        self.assertEqual(command.count("-i"), 2)

    def test_verifier_data_never_enters_public_trace(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            job = root / "job"
            trial = job / "extract-elf__abc"
            agent_dir = trial / "agent"
            agent_dir.mkdir(parents=True)
            trajectory = {
                "schema_version": "ATIF-v1.6",
                "session_id": "s",
                "agent": {"name": "kira", "version": "1"},
                "steps": [
                    {
                        "step_id": 1,
                        "source": "user",
                        "message": "perform the task",
                    },
                    {
                        "step_id": 2,
                        "source": "agent",
                        "message": "I will check the artifact",
                        "tool_calls": [
                            {
                                "tool_call_id": "c1",
                                "function_name": "bash_command",
                                "arguments": {"keystrokes": "file /root/out"},
                            },
                            {
                                "tool_call_id": "c2",
                                "function_name": "mark_task_complete",
                                "arguments": {},
                            },
                        ],
                        "observation": {"results": [{"content": "ELF 64-bit"}]},
                    },
                ],
            }
            raw = json.dumps(trajectory, separators=(",", ":"))
            (agent_dir / "trajectory.json").write_text(raw, encoding="utf-8")
            (trial / "result.json").write_text(
                json.dumps(
                    {
                        "task_name": "extract-elf",
                        "agent_result": {
                            "n_input_tokens": 10,
                            "metadata": {"n_episodes": 2},
                        },
                        "verifier_result": {
                            "rewards": {"reward": 1},
                            "stdout": "TOP_SECRET_GOLD_OUTPUT",
                        },
                    }
                ),
                encoding="utf-8",
            )
            public = root / "public"
            observations = collect_harbor_observations(
                job, "baseline_kira", ["extract-elf"], public
            )

            self.assertEqual(len(observations), 1)
            self.assertEqual(observations[0].reward, 1.0)
            self.assertEqual(observations[0].raw_trace_path.read_text(), raw)
            all_public = "\n".join(path.read_text() for path in public.iterdir())
            self.assertNotIn("TOP_SECRET_GOLD_OUTPUT", all_public)
            self.assertNotIn("verifier_result", all_public)
            evidence = json.loads(observations[0].evidence_path.read_text())
            self.assertEqual(evidence["task_complete_calls"], 1)
            self.assertEqual(
                evidence["commands_and_observations"][0]["observations"], ["ELF 64-bit"]
            )


class AuditTests(unittest.TestCase):
    def test_minimal_agent_is_allowed(self):
        source = """
class AgentHarness(Terminus2):
    async def run(self, instruction, environment, context):
        return await super().run(instruction, environment, context)
"""
        self.assertEqual(audit_source(source), [])

    def test_model_budget_and_gold_access_are_rejected(self):
        source = """
class AgentHarness(Terminus2):
    def bad(self):
        self._model_name = 'other-model'
        self._max_episodes = 999
        return '/logs/verifier/reward.txt'
"""
        rules = {item["rule"] for item in audit_source(source)}
        self.assertIn("FROZEN_SOLVER", rules)
        self.assertIn("LABEL_FREE", rules)

    def test_task_name_hardcoding_is_rejected(self):
        source = """
class AgentHarness(Terminus2):
    hint = 'special handling for extract-elf'
"""
        rules = {
            item["rule"]
            for item in audit_source(source, forbidden_task_names=("extract-elf",))
        }
        self.assertIn("NO_TASK_HARDCODING", rules)


class JudgeTests(unittest.TestCase):
    def test_provenance_records_exact_parent_and_diff(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            agents = root / "agents"
            agents.mkdir()
            (agents / "baseline.py").write_text("value = 1\n", encoding="utf-8")
            (agents / "challenger.py").write_text("value = 2\n", encoding="utf-8")
            lineage_path = root / "branch_lineage.jsonl"
            lineage_path.write_text(
                json.dumps(
                    {
                        "batch": 0,
                        "round": 0,
                        "branch": 0,
                        "parent": "baseline",
                        "proposed": "challenger",
                        "active": "challenger",
                        "fell_back": False,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            task = {
                "candidates": ["baseline", "challenger"],
                "incumbent": "baseline",
                "run_dir": str(root),
                "tag": "b0",
                "lineage_path": str(lineage_path),
            }
            with patch("tthe_proposer.AGENTS_DIR", agents):
                provenance_path = _prepare_judge_provenance(task)

            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
            by_name = {
                row["candidate"]: row for row in provenance["candidates"]
            }
            self.assertEqual(by_name["baseline"]["role"], "incumbent")
            self.assertIsNone(by_name["baseline"]["immediate_parent"])
            self.assertEqual(
                by_name["challenger"]["immediate_parent"], "baseline"
            )
            diff = Path(
                by_name["challenger"]["immediate_parent_diff_path"]
            ).read_text(encoding="utf-8")
            self.assertIn("-value = 1", diff)
            self.assertIn("+value = 2", diff)

    def test_judge_remains_model_decided_without_hard_gates(self):
        task = {
            "candidates": ["baseline", "challenger"],
            "incumbent": "baseline",
            "trace_dir": "/tmp/traces",
            "choice_path": "/tmp/judge.txt",
        }
        prompt = _judge_prompt(task, Path("/tmp/provenance.json"))
        self.assertIn("including the incumbent", prompt)
        self.assertIn("controller has not filtered or ranked", prompt)
        self.assertIn("never automatic eligibility or\ndisqualification rules", prompt)
        self.assertIn("DECISION SELF-CHECK", prompt)
        self.assertIn("mechanically recorded immediate parent", prompt)


class SafeDockerTests(unittest.TestCase):
    def config(self):
        from safe_docker import NETWORK
        return {
            "services": {"main": {
                "image": "cached-image", "pull_policy": "never", "networks": {"tthe": {}},
                "volumes": [{"type": "bind", "source": "/tmp/trial/agent", "target": "/logs/agent"}],
                "deploy": {"resources": {"limits": {"cpus": "1", "memory": "2G"}}},
            }},
            "networks": {"tthe": {"external": True, "name": NETWORK}},
        }

    def test_isolated_config_is_allowed(self):
        from safe_docker import validate_config
        validate_config(self.config(), Path("/tmp/trial"))

    def test_host_access_privilege_pulls_and_external_mounts_are_rejected(self):
        from safe_docker import validate_config
        for key, value in (
            ("network_mode", "host"), ("network_mode", "bridge"), ("privileged", True),
            ("cap_add", ["NET_ADMIN"]), ("pid", "host"), ("ipc", "host"),
            ("ports", ["8000:8000"]), ("pull_policy", "always"),
            ("build", {"context": "."}),
            ("volumes", [{"type": "bind", "source": "/var/run/docker.sock", "target": "/docker.sock"}]),
        ):
            with self.subTest(key=key, value=value):
                config = copy.deepcopy(self.config())
                config["services"]["main"][key] = value
                with self.assertRaises(RuntimeError):
                    validate_config(config, Path("/tmp/trial"))


class ResumeTests(unittest.TestCase):
    def setUp(self):
        import resume_continue_b8 as resume
        self.resume = resume
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.run = self.root / "run"
        self.run.mkdir()
        (self.run / "opt_log.jsonl").write_text("")
        self.private_patch = patch.object(resume, "PRIVATE", self.root / "private")
        self.run_patch = patch.object(resume, "RUN", self.run)
        self.private_patch.start()
        self.run_patch.start()

    def tearDown(self):
        self.private_patch.stop()
        self.run_patch.stop()
        self.temp.cleanup()

    def trace(self, directory, task, exception=None):
        directory.mkdir(parents=True, exist_ok=True)
        stem = f"candidate__{task}__a0"
        (directory / f"{stem}.evidence.json").write_text(json.dumps({
            "task_name": task, "status": {"exception_type": exception}}))
        (directory / f"{stem}.raw_trajectory.json").write_text(json.dumps({"steps": [{"step_id": 1}]}))
        (directory / f"{stem}.md").write_text("public trace")
        return self.resume.bridge.TrialObservation(
            task, 0, 1.0, {"exception_type": exception},
            directory / f"{stem}.raw_trajectory.json", directory / f"{stem}.md",
            directory / f"{stem}.evidence.json",
        )

    def test_task_failure_is_reused_but_cancellation_is_not(self):
        directory = self.root / "traces"
        self.trace(directory, "failed", "RewardFileNotFoundError")
        self.trace(directory, "cancelled", "CancelledError")
        self.assertTrue(self.resume.reusable("candidate", "failed", directory))
        self.assertFalse(self.resume.reusable("candidate", "cancelled", directory))

    def test_only_missing_task_is_run_and_reward_stays_private(self):
        directory = self.root / "traces"
        self.trace(directory, "cached")
        original = (directory / "candidate__cached__a0.raw_trajectory.json").read_bytes()

        def fake_observe(candidate, import_path, tasks, attempt_dir, **kwargs):
            self.assertEqual(tasks, ["missing"])
            self.assertEqual(kwargs["concurrency"], 1)
            return [self.trace(attempt_dir, "missing")]

        with patch.object(self.resume.bridge, "observe_candidate", side_effect=fake_observe) as observe:
            result = self.resume.observe("candidate", ["cached", "missing"], directory)
            self.assertEqual(observe.call_count, 1)
        self.assertEqual(len(result), 1)
        self.assertEqual((directory / "candidate__cached__a0.raw_trajectory.json").read_bytes(), original)
        self.assertEqual(self.resume.cached_observation("candidate", "missing", directory).reward, 1.0)
        self.assertNotIn("reward", (directory / "candidate__missing__a0.evidence.json").read_text())
        with patch.object(self.resume.bridge, "observe_candidate") as observe:
            self.resume.observe("candidate", ["cached", "missing"], directory)
            observe.assert_not_called()

    def test_duplicate_or_partial_score_cannot_be_committed(self):
        observation = self.trace(self.root / "traces", "task")
        with self.assertRaises(RuntimeError):
            self.resume.append_score(10, ["task", "missing"], "candidate", [observation])
        self.resume.append_score(10, ["task"], "candidate", [observation])
        with self.assertRaises(RuntimeError):
            self.resume.append_score(10, ["task"], "candidate", [observation])
        self.assertEqual(len((self.run / "opt_log.jsonl").read_text().splitlines()), 1)


if __name__ == "__main__":
    unittest.main()
