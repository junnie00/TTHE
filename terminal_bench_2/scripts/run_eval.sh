#!/usr/bin/env bash
# Usage: scripts/run_eval.sh <agent_import_path> [hard|full] [runs] [n_concurrent] [extra_harbor_flags...]

set -euo pipefail

AGENT_IMPORT_PATH="${1:?Usage: $0 <agent_import_path> [task_set] [runs] [n_concurrent] [extra_harbor_flags...]}"
TASK_SET="${2:-full}"
RUNS="${3:-2}"
N_CONCURRENT="${4:-50}"
shift "$(( $# < 4 ? $# : 4 ))"
EXTRA_FLAGS=("$@")

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

# Load local .env (override shell environment)
if [[ -f "$REPO_DIR/.env" ]]; then
    set -a
    # shellcheck source=/dev/null
    source "$REPO_DIR/.env"
    set +a
fi

MODEL="${HARBOR_MODEL:-anthropic/claude-opus-4-6}"

# 30-task hard subset for cheaper development and debugging loops.
HARD_TASKS=(
    bn-fit-modify cancel-async-tasks circuit-fibsqrt configure-git-webserver
    dna-assembly extract-moves-from-video feal-differential-cryptanalysis
    feal-linear-cryptanalysis fix-code-vulnerability fix-ocaml-gc
    gpt2-codegolf install-windows-3.11 llm-inference-batching-scheduler
    make-doom-for-mips make-mips-interpreter mcmc-sampling-stan
    model-extraction-relu-logits password-recovery path-tracing
    path-tracing-reverse polyglot-rust-c protein-assembly regex-chess
    sam-cell-seg sparql-university torch-pipeline-parallelism
    torch-tensor-parallelism train-fasttext video-processing write-compressor
)

TASK_FLAGS=()
case "$TASK_SET" in
    hard)
        for t in "${HARD_TASKS[@]}"; do TASK_FLAGS+=(-i "$t"); done ;;
    full)
        ;;  # no task flags: harbor runs the full dataset
    *)
        echo "Unknown task_set '$TASK_SET'. Use: hard | full" >&2; exit 1 ;;
esac

echo "agent:       $AGENT_IMPORT_PATH"
echo "task_set:    $TASK_SET"
echo "model:       $MODEL"
echo "concurrent:  $N_CONCURRENT"
echo "runs:        $RUNS"
echo ""

# Prefer an outer wall-clock timeout when GNU timeout is available.
TIMEOUT_CMD=()
if command -v timeout >/dev/null 2>&1; then
    TIMEOUT_CMD=(timeout --signal=TERM --kill-after=60 2h)
elif command -v gtimeout >/dev/null 2>&1; then
    TIMEOUT_CMD=(gtimeout --signal=TERM --kill-after=60 2h)
fi

CMD=(
    uv run harbor run
    --agent-import-path "$AGENT_IMPORT_PATH"
    -d "terminal-bench@2.0"
    -m "$MODEL"
    -e runloop
    -n "$N_CONCURRENT"
    --n-attempts "$RUNS"
)
if [[ ${#TASK_FLAGS[@]} -gt 0 ]]; then
    CMD+=("${TASK_FLAGS[@]}")
fi
if [[ ${#EXTRA_FLAGS[@]} -gt 0 ]]; then
    CMD+=("${EXTRA_FLAGS[@]}")
fi

if [[ ${#TIMEOUT_CMD[@]} -gt 0 ]]; then
    "${TIMEOUT_CMD[@]}" "${CMD[@]}"
else
    "${CMD[@]}"
fi
