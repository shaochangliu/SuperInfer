#!/usr/bin/env bash

set -euo pipefail

REPO_DIR="/home/sliu27/baseline/SuperInfer"

LOG_DIR="$REPO_DIR/logs/e2e_perf"
VENV_ACTIVATE="$REPO_DIR/.venv/bin/activate"
CURRENT_SUPERINFER_PGID_FILE="$LOG_DIR/current_superinfer.pgid"

cd "$REPO_DIR"
mkdir -p "$LOG_DIR"

if [[ ! -f "$VENV_ACTIVATE" ]]; then
    echo "[Error] Expected venv activation script at $VENV_ACTIVATE" >&2
    exit 1
fi

source "$VENV_ACTIVATE"

export TORCH_CUDA_ARCH_LIST=9.0
export VLLM_USE_V1=1
export VLLM_USE_FLASHINFER_SAMPLER=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

GPU_MEMORY_UTILIZATION=0.98
SWAP_SPACE=200
PROACTIVE_SWAP_BUDGET=2400      # AE setting
MAX_NUM_SEQS=16384              # AE setting
MAX_NUM_BATCHED_TOKENS=16384    # AE setting
NB=16                           # Castor batch count used only to derive requests
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

watch_for_fatal_engine_error() {
    local pgid=$1
    local log_file=$2

    while kill -0 "-$pgid" 2>/dev/null; do
        if grep -qE \
            'No available memory for the cache blocks|Engine core initialization failed|EngineCore failed to start' \
            "$log_file"; then
            echo "[Error] Fatal engine initialization error detected; stopping SuperInfer process group $pgid."
            kill -TERM "-$pgid" 2>/dev/null || true
            sleep 5
            kill -KILL "-$pgid" 2>/dev/null || true
            return
        fi
        sleep 5
    done
}

trap cleanup_on_exit EXIT
trap 'exit 124' TERM
trap 'exit 130' INT
timeout_self &
WATCHDOG_PID=$!

# The 0.98 GPU budget is about 93.66 GiB on this GH200. These offload
# amounts leave approximately 10 GiB in that budget for non-weight memory.
# NPL is Castor metadata; baseline request count is NPL * NB.
# Format: "ModelName:HFModelPath:PP:TG:NPL:CPUOffloadGiB"
runs=(
    "Llama-3.1-70B:/tmp/models/Llama-3.1-70B:512:128:128:48"
    "Llama-3.1-70B:/tmp/models/Llama-3.1-70B:256:256:128:48"
    "Llama-3.1-70B:/tmp/models/Llama-3.1-70B:128:512:128:48"

    "Qwen2MoE:/tmp/models/Qwen2-57B-A14B:1024:256:128:24"
    "Qwen2MoE:/tmp/models/Qwen2-57B-A14B:512:512:128:24"
    "Qwen2MoE:/tmp/models/Qwen2-57B-A14B:256:1024:128:24"
)

run_superinfer_cmd() {
    local name=$1; local model_path=$2; local npp=$3; local ntg=$4
    local npl=$5; local cpu_offload_gb=$6

    local num_prompts=$(( npl * NB ))
    local max_model_len=$(( npp + ntg ))
    local log_file="${LOG_DIR}/${name}.log"
    local json_file="${LOG_DIR}/${name}.json"

    echo "  -> Executing: $name"
    echo "     Requests: NPL ${npl} x NB ${NB} = ${num_prompts}"
    echo "     CPU weight offload: ${cpu_offload_gb} GiB"
    local full_cmd=(
        numactl --cpunodebind=0 --membind=0
        python benchmarks/benchmark_throughput.py
        --backend vllm
        --async-engine
        --model "$model_path"
        --dtype bfloat16
        --num-prompts "$num_prompts"
        --input-len "$npp"
        --output-len "$ntg"
        --max-model-len "$max_model_len"
        --cpu-offload-gb "$cpu_offload_gb"
        --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
        --swap-space "$SWAP_SPACE"
        --proactive-swap-budget "$PROACTIVE_SWAP_BUDGET"
        --max-num-seqs "$MAX_NUM_SEQS"
        --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
        --swapper-block-first
        --pin-memory-fix
        --prefix-cache-fix
        --no-enable-prefix-caching
        --output-json "$json_file"
        --record-finish-time-distribution
    )

    printf "     Command:"
    printf " %q" "${full_cmd[@]}"
    printf "\n"

    cd "$REPO_DIR"
    setsid "${full_cmd[@]}" > "$log_file" 2>&1 &
    local superinfer_pid=$!
    CURRENT_SUPERINFER_PGID=$superinfer_pid
    printf "%s\n" "$CURRENT_SUPERINFER_PGID" > "$CURRENT_SUPERINFER_PGID_FILE"

    watch_for_fatal_engine_error "$superinfer_pid" "$log_file" &
    local error_monitor_pid=$!

    set +e
    wait "$superinfer_pid"
    local status=$?
    kill "$error_monitor_pid" 2>/dev/null || true
    wait "$error_monitor_pid" 2>/dev/null || true
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
    IFS=':' read -r M_NAME HF_PATH NPP NTG NPL CPU_OFFLOAD_GB <<< "$entry"

    COUNT=$((COUNT + 1))
    echo "=================================================="
    echo "[$COUNT/$TOTAL] Model: $M_NAME | PP: $NPP | TG: $NTG | NPL: $NPL | NB: $NB | Requests: $((NPL * NB)) | CPU offload: ${CPU_OFFLOAD_GB} GiB"
    echo "=================================================="

    PREFIX="${M_NAME}_PP${NPP}_TG${NTG}_NPL${NPL}_NB${NB}_REQ$((NPL * NB))_CPUOFFLOAD${CPU_OFFLOAD_GB}GB"
    run_superinfer_cmd "${PREFIX}_SuperInfer" \
        "$HF_PATH" "$NPP" "$NTG" "$NPL" "$CPU_OFFLOAD_GB"

done

echo "=================================================="
echo "SuperInfer benchmarks completed. Logs saved in $LOG_DIR."
