#!/usr/bin/env bash

set -euo pipefail

REPO_DIR="/scratch/sliu27/SuperInfer"

LOG_DIR="$REPO_DIR/logs"
VENV_ACTIVATE="$REPO_DIR/.venv/bin/activate"
CURRENT_SUPERINFER_PGID_FILE="$LOG_DIR/current_superinfer.pgid"

cd "$REPO_DIR"
mkdir -p "$LOG_DIR"

if [[ ! -f "$VENV_ACTIVATE" ]]; then
    echo "[Error] Expected venv activation script at $VENV_ACTIVATE" >&2
    exit 1
fi

source "$VENV_ACTIVATE"

export CUDA_HOME=/scratch/sliu27/cuda-12.8.1
export CUDA_PATH=$CUDA_HOME
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST=9.0

GPU_MEM_UTILIZATION=0.24 # 23GB
SWAP_SPACE=200
PROACTIVE_SWAP_BUDGET=400       # proportional to AE setting
MAX_NUM_SEQS=16384              # AE setting
MAX_NUM_BATCHED_TOKENS=16384    # AE setting
NB=16
SCRIPT_TIMEOUT_SECONDS=$((6 * 60 * 60))
CURRENT_SUPERINFER_PGID=""
WATCHDOG_PID=""

cleanup_current_superinfer() {
    local pgid="${CURRENT_SUPERINFER_PGID:-}"
    if [[ -z "$pgid" && -f "$CURRENT_SUPERINFER_PGID_FILE" ]]; then
        pgid=$(<"$CURRENT_SUPERINFER_PGID_FILE")
    fi

    if [[ -z "$pgid" ]]; then
        return
    fi

    if kill -0 "-$pgid" 2>/dev/null; then
        echo "[Cleanup] Stopping SuperInfer process group $pgid"
        kill -TERM "-$pgid" 2>/dev/null || true
        sleep 10
        kill -KILL "-$pgid" 2>/dev/null || true
    fi
    rm -f "$CURRENT_SUPERINFER_PGID_FILE"
}

cleanup_on_exit() {
    local status=$?
    if [[ -n "${WATCHDOG_PID:-}" ]]; then
        kill "$WATCHDOG_PID" 2>/dev/null || true
    fi
    cleanup_current_superinfer
    exit "$status"
}

timeout_self() {
    sleep "$SCRIPT_TIMEOUT_SECONDS"
    echo "[Timeout] run_Super.sh exceeded ${SCRIPT_TIMEOUT_SECONDS}s; terminating current SuperInfer run and exiting."
    cleanup_current_superinfer
    kill -TERM "$$" 2>/dev/null || true
}

trap cleanup_on_exit EXIT
trap 'exit 124' TERM
trap 'exit 130' INT
timeout_self &
WATCHDOG_PID=$!

# Format: "ModelName:HFModelPath:NPP:NTG:NPL"
# SuperInfer num-prompts is NPL * NB to align with llama.cpp's NPL/NB setup.
runs=(
    # "llama3.1-8b:/scratch/sliu27/models/HF/llama3.1-8b:1024:256:48"
    # "llama3.1-8b:/scratch/sliu27/models/HF/llama3.1-8b:256:1024:48"
    # "llama3.1-8b:/scratch/sliu27/models/HF/llama3.1-8b:512:512:60"

    # "llama2-7b:/scratch/sliu27/models/HF/llama2-7b:1024:256:14"
    "llama2-7b:/scratch/sliu27/models/HF/llama2-7b:256:1024:14"
    "llama2-7b:/scratch/sliu27/models/HF/llama2-7b:512:512:18"

    "MPT-7b:/scratch/sliu27/models/HF/MPT-7b:1024:256:14"
    "MPT-7b:/scratch/sliu27/models/HF/MPT-7b:256:1024:14"
    "MPT-7b:/scratch/sliu27/models/HF/MPT-7b:512:512:18"
)

run_superinfer_cmd() {
    local name=$1; local model_path=$2; local npp=$3; local ntg=$4; local npl=$5; local nb=$6

    local num_prompts=$(( npl * nb ))
    local max_model_len=$(( npp + ntg ))
    local log_file="${LOG_DIR}/${name}.log"
    local json_file="${LOG_DIR}/${name}.json"

    echo "  -> Executing: $name"
    local full_cmd=(
        python benchmarks/benchmark_throughput.py
        --backend vllm
        --async-engine
        --model "$model_path"
        --num-prompts "$num_prompts"
        --input-len "$npp"
        --output-len "$ntg"
        --max-model-len "$max_model_len"
        --gpu-memory-utilization "$GPU_MEM_UTILIZATION"
        --swap-space "$SWAP_SPACE"
        --proactive-swap-budget "$PROACTIVE_SWAP_BUDGET"
        --max-num-seqs "$MAX_NUM_SEQS"
        --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
        --swapper-block-first
        --pin-memory-fix
        --prefix-cache-fix
        --no-enable-prefix-caching
        --output-json "$json_file"
    )

    printf "     Command:"
    printf " %q" "${full_cmd[@]}"
    printf "\n"

    cd "$REPO_DIR"
    export VLLM_USE_V1=1
    export VLLM_USE_FLASHINFER_SAMPLER=1
    export VLLM_WORKER_MULTIPROC_METHOD=spawn
    setsid "${full_cmd[@]}" > "$log_file" 2>&1 &
    local superinfer_pid=$!
    CURRENT_SUPERINFER_PGID=$superinfer_pid
    printf "%s\n" "$CURRENT_SUPERINFER_PGID" > "$CURRENT_SUPERINFER_PGID_FILE"

    set +e
    wait "$superinfer_pid"
    local status=$?
    set -e
    CURRENT_SUPERINFER_PGID=""
    rm -f "$CURRENT_SUPERINFER_PGID_FILE"

    if (( status != 0 )); then
        echo "    [Error] $name failed. See $log_file"
        return "$status"
    fi

    sleep 5
}

TOTAL=${#runs[@]}
COUNT=0

for entry in "${runs[@]}"; do
    IFS=':' read -r M_NAME HF_PATH NPP NTG NPL <<< "$entry"

    COUNT=$((COUNT + 1))
    echo "=================================================="
    echo "[$COUNT/$TOTAL] Model: $M_NAME | NPP: $NPP | NTG: $NTG | NPL: $NPL | NB: $NB"
    echo "=================================================="

    PREFIX="${M_NAME}_NPP${NPP}_NTG${NTG}_NPL${NPL}_NB${NB}"
    run_superinfer_cmd "${PREFIX}_SuperInfer" "$HF_PATH" "$NPP" "$NTG" "$NPL" "$NB"

done

echo "=================================================="
echo "SuperInfer benchmarks completed. Logs saved in $LOG_DIR."
