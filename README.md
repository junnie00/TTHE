# TTHE — Test-Time Harness Evolution

TTHE optimizes an agent's **executable harness** *during evaluation*, using only unlabeled execution
traces from the test stream. Model weights stay frozen; gold labels never enter the loop (they are used
only afterwards, to report accuracy). On each unlabeled batch the loop evolves **G fixed branches** over
**R rounds** — each branch's agentic proposer edits its own parent harness — and an agentic judge commits
one final harness that carries forward to the next batch.

This repository contains the implementation across several execution-grounded domains.

## Layout

```
TTHE/
├── ase/                # SHARED infra: OpenAI-compatible LLM client, SQLite exec, dataset loaders
├── text_to_sql/        # BIRD (Text-to-SQL)
├── livecodebench/      # LiveCodeBench (competitive programming)
├── ds1000/             # DS-1000 (data-science coding)
├── swe/                # SWE-bench Verified (real-world software engineering; via mini-swe-agent)
├── config.example.yaml # copy to config.yaml and fill in your endpoint
├── requirements.txt
└── LICENSE             # MIT
```

Each domain folder holds its own `*_optimize.py` (the TTHE loop), `*_proposer.py` (agentic proposers +
judge), `*_bridge.py` (data/exec/LLM bridge), `harness_base.py` (the harness interface), and
`agents/` with the **seed harnesses** (`bare.py` = one greedy shot; `react.py` = a ReAct loop with
execution feedback). All four domains share the same fixed-branch search, the same three proposer roles
(conservative-repair / independent-exploration / adversarial-audit), and, where applicable, a round-trip
(back-translation) proxy signal.

> The agentic proposer/judge run through the **Claude Code CLI** wrapper in `text_to_sql/claude_wrapper.py`,
> which every domain imports — so the `text_to_sql/` package must stay present even when running another
> domain.

## Install

```bash
pip install -r requirements.txt
```

The proposer and judge run inside the **Claude Code CLI** (`claude`), used purely as an agentic coding
scaffold — its model calls are routed to the same frozen backbone as the solver (via
`ANTHROPIC_BASE_URL` / `ANTHROPIC_AUTH_TOKEN`). Install it separately:
<https://docs.claude.com/claude-code>.

## Configure

```bash
cp config.example.yaml config.yaml     # config.yaml is .gitignored — your key never gets committed
export OPENAI_API_KEY=sk-...           # the env var named by llm.api_key_env in config.yaml
```

`config.yaml` sets the frozen solver: `llm.base_url` / `llm.api_key_env` (any OpenAI-compatible endpoint)
and `llm.solver_model`. Override the config path with `TTHE_CONFIG=/path/to/config.yaml`. Run every
command **from the repo root** with `PYTHONPATH=.` so the domain packages and the shared `ase` resolve.

## Per-domain: data + run

**Text-to-SQL (BIRD).** Offline demo (built-in tiny DB, no download):
```bash
PYTHONPATH=. python -m text_to_sql.optimize --db demo --cap 5 --max-rounds 3
```
For BIRD, set `dataset.name: bird` + `dataset.bird_root` in `config.yaml` (dir with `dev.json` +
`dev_databases/<db_id>/<db_id>.sqlite`). To evolve on the hard slice, stream it with
`--cross-set text_to_sql/slices/genuine_hard50.json` and export `BIRD_DEV_FILE=mini_dev.json` — the
slice's `[db, idx]` entries are indices into `mini_dev.json`, so it must be the active dev file.

**LiveCodeBench.** Data is pulled from the HuggingFace hub (`livecodebench/code_generation_lite`, default
`test6`). Baseline then evolution (the hard slice is `livecodebench/slices/hard60.json`):
```bash
PYTHONPATH=. python -m livecodebench.lcb_bare  livecodebench/slices/hard60.json  bare_hard
PYTHONPATH=. python -m livecodebench.lcb_optimize --group 3 --max-rounds 3 ...
```

**DS-1000.** Data from HuggingFace (`xlangai/DS-1000`, split `test`). Generated snippets execute against
`numpy`/`pandas`/`scipy`/`scikit-learn` (install those). Baseline then evolution (hard slice
`ds1000/slices/hard50.json`):
```bash
PYTHONPATH=. python -m ds1000.ds1000_bare  ds1000/slices/hard50.json  bare_hard
PYTHONPATH=. python -m ds1000.ds1000_optimize --group 3 --max-rounds 3 ...
```

**SWE-bench Verified** (heaviest — needs extra setup):
- `pip install mini-swe-agent swebench litellm`
- **Docker** (each rollout and the gold scoring run in per-instance containers)
- dataset `princeton-nlp/SWE-Bench_Verified` (auto-pulled from HuggingFace)
```bash
PYTHONPATH=. python -m swe.swe_optimize --group 3 --max-rounds 3 ...
```
The solver runs as a frozen `mini-swe-agent` rollout; gold scoring calls the official
`python -m swebench.harness.run_evaluation`.

## Notes

- **Evaluation slices.** Each domain's exact hard slice (small ID lists) lives in `<domain>/slices/` —
  `text_to_sql/slices/genuine_hard50.json`, `livecodebench/slices/hard60.json`,
  `ds1000/slices/hard50.json`, `swe/slices/hard40.json`. The full benchmarks themselves are fetched from
  their public sources (HuggingFace / BIRD), not vendored here.
- **claw-eval.** `claweval/slices/headroom30.json` lists the 30-task headroom slice (the 30 lowest-bare
  tasks of the 112 local-only agentic tasks) used for the claw-eval result. The loop code for this domain
  is **not** included here: it drives [claw-eval](https://github.com/claw-eval/claw-eval) through a large
  external agent runtime, so only the slice is published for reference.
- **Label-free loop.** Gold is used only to *report* accuracy after a harness is committed; it never
  reaches the harness, proposer, judge, or traces.
- **`config.yaml`, `logs/`, `runs/`, evolved `agents/cand_*.py`, and caches are gitignored** — they hold
  secrets, large traces, or regenerated run outputs. Only the seed harnesses (`bare.py` / `react.py`) are
  tracked.
- **`swe/` is deliberately not named `swebench`** to avoid shadowing the official `swebench` pip package
  used for gold scoring.
