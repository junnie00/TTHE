# Terminal-Bench 2

Terminal-Bench 2 reference experiment for Meta-Harness. The default search config in this release uses Harbor on the full 89-task TB2 dataset, with 2 search trials per task on Opus 4.6.

This directory now contains two distinct experiment drivers:

- `tthe_optimize.py` is the trace-driven TTHE implementation aligned with the formal TTHE repository. It uses fixed proposer branches, a label-free agentic judge, an incumbent rollback gate, and post-commit reporting of Harbor reward.
- `meta_harness.py` is the earlier reward-frontier experiment retained for reproducibility. It reads verifier pass rates during search and is not the formal label-free TTHE loop.

## Quick Start

Install:

```bash
cd reference_examples/terminal_bench_2
uv sync
```

Read setup details first:

```bash
sed -n '1,200p' SETUP.md
```

Start with a cheap smoke check:

```bash
uv run bash scripts/run_eval.sh agents.baseline_kira:AgentHarness full 1 1 -i extract-elf
```

Validate the TTHE wiring without starting a Harbor job or proposer:

```bash
uv run python tthe_optimize.py --validate-only
```

Run the smallest TTHE smoke evolution (two agent rollouts on one task: incumbent plus one child):

```bash
uv run python tthe_optimize.py \
  --tasks slices/smoke.json \
  --batch-size 1 --group 1 --max-rounds 1 \
  --concurrent 1 --max-turns 8 --run-name smoke
```

`--max-turns` is a controller-fixed solver budget shared by the incumbent and
every candidate. The default is `100`; the smoke command uses `8` only to check
the complete pipeline cheaply, so its task reward is not a performance estimate.

Run the published 30-task development slice only after the smoke path works:

```bash
uv run python tthe_optimize.py \
  --tasks slices/hard30.json \
  --batch-size 5 --group 2 --max-rounds 2 \
  --concurrent 5 --run-name hard30
```

`slices/full89.json` contains the task identities at the dataset commit recorded in `SETUP.md`.

When trying a new idea, validate it on the cheaper 30-task `hard` subset before paying for a full default search:

```bash
uv run bash scripts/run_eval.sh agents.baseline_kira:AgentHarness hard 1 50
```

Run one evolve iteration with the default full-dataset search config:

```bash
uv run python meta_harness.py --iterations 1
```

Pass `--full-eval` if you also want the optional 5-trial winner pass on the full dataset.

## Repro And Troubleshooting

- The shipped `runloop` path requires both `ANTHROPIC_API_KEY` and `RUNLOOP_API_KEY`.
- The default paper-style config in this release is Opus 4.6, `full`, `89` tasks, `2` search trials, and concurrency `50`.
- Anthropic API tier matters for both speed and failure rate.
- Sharing the same Anthropic API key with other active projects can make runs substantially slower.
- Many failures at higher concurrency are timeout failures caused by insufficient API throughput, not necessarily reasoning failures.
- The recommended bring-up order is `extract-elf`, then `hard`, then the full default run.

## Key Files

- `.claude/skills/meta-harness-terminal-bench-2/SKILL.md`: proposer prior used by `meta_harness.py`.
- `agents/`: baseline agents and the write target for generated candidates.
- `prompt-templates/terminus-kira.txt`: prompt template used by `baseline_kira.py`.
- `tthe_optimize.py`: formal fixed-branch `OBSERVE -> GENERATE -> PICK -> SCORE` loop.
- `tthe_bridge.py`: Harbor runner and verifier-isolating trace exporter.
- `tthe_proposer.py`: branch proposer and label-free judge.
- `audit_agent.py`: frozen-model, fixed-budget, and label-free invariant audit.
- `slices/`: explicit smoke, hard30, and full89 task streams.

## TTHE Evidence And Gold Isolation

For each candidate/task, the TTHE loop publishes:

- `*.raw_trajectory.json`: byte-exact Harbor ATIF agent trajectory;
- `*.md`: bounded reading view of that trajectory;
- `*.evidence.json`: factual commands, observations, completion calls, and operational metadata.

There is no weighted proxy score. Proposers and the judge reason directly from task-grounded terminal evidence. Before judging, the controller records exact branch lineage and immediate-parent source diffs as factual evidence, without filtering or ranking candidates. The same judge first builds a per-task evidence matrix, then checks the provisional winner against its strongest counterevidence and the incumbent before choosing any candidate, including the incumbent. Exceptions and completion calls remain contextual evidence rather than hard eligibility rules.

Harbor couples agent execution and verification in one trial, so `tthe_bridge.py` runs each job in a temporary controller-only directory, copies out only agent files, keeps candidate rewards in controller memory, and deletes the Harbor job directory before a proposer starts. Only the committed harness's rewards are serialized after `PICK`; unselected rewards are discarded.

## Runtime And Cost

With Opus 4.6 and a high-tier API key, the default 89x2 search run at concurrency `50` takes about 4-6 hours and costs roughly $500 _per iteration_. The recommended bring-up path is `extract-elf`, then `hard`, then the full default run.

This setup is sensitive to concurrency and Anthropic API throughput. API tier matters, and sharing the same API key with other active projects can make runs much slower. Many failures at higher concurrency are timeout failures caused by insufficient API throughput at the chosen setting.
