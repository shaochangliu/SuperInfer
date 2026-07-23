#!/usr/bin/env bash

set -euo pipefail

REPO_DIR="/home/sliu27/baseline/SuperInfer"

LOG_DIR="$REPO_DIR/logs_largerPP"
VENV_ACTIVATE="$REPO_DIR/.venv/bin/activate"
CURRENT_SUPERINFER_PGID_FILE="$LOG_DIR/current_superinfer.pgid"
CURRENT_RESERVER_PID_FILE="$LOG_DIR/current_cuda_memory_reserver.pid"

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

GPU_MEMORY_LIMIT_GB=23
VLLM_GPU_MEMORY_HEADROOM=0.98
CUDA_RESERVER_DEVICE=0
CUDA_RESERVER_SRC="$REPO_DIR/tools/cuda_memory_reserver.cu"
CUDA_RESERVER_BIN="$REPO_DIR/tools/cuda_memory_reserver"
SWAP_SPACE=200
PROACTIVE_SWAP_BUDGET=400       # proportional to AE setting
MAX_NUM_SEQS=16384              # AE setting
MAX_NUM_BATCHED_TOKENS=16384    # AE setting
NB=16
SCRIPT_TIMEOUT_SECONDS=$((6 * 60 * 60))
CURRENT_SUPERINFER_PGID=""
CURRENT_RESERVER_PID=""
CURRENT_RESERVER_READY_FILE=""
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

cleanup_current_reserver() {
    local pid="${CURRENT_RESERVER_PID:-}"
    if [[ -z "$pid" && -f "$CURRENT_RESERVER_PID_FILE" ]]; then
        pid=$(<"$CURRENT_RESERVER_PID_FILE")
    fi

    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
        echo "[Cleanup] Stopping CUDA memory reserver $pid"
        kill -TERM "$pid" 2>/dev/null || true
        sleep 2
        wait "$pid" 2>/dev/null || true
        kill -KILL "$pid" 2>/dev/null || true
    fi
    CURRENT_RESERVER_PID=""
    rm -f "$CURRENT_RESERVER_PID_FILE"
    if [[ -n "${CURRENT_RESERVER_READY_FILE:-}" ]]; then
        rm -f "$CURRENT_RESERVER_READY_FILE"
        CURRENT_RESERVER_READY_FILE=""
    fi
}

cleanup_on_exit() {
    local status=$?
    if [[ -n "${WATCHDOG_PID:-}" ]]; then
        kill "$WATCHDOG_PID" 2>/dev/null || true
    fi
    cleanup_current_superinfer
    cleanup_current_reserver
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

build_cuda_memory_reserver() {
    if [[ ! -x "$CUDA_RESERVER_BIN" || "$CUDA_RESERVER_SRC" -nt "$CUDA_RESERVER_BIN" ]]; then
        echo "[Build] Compiling CUDA memory reserver"
        "$CUDA_HOME/bin/nvcc" -O2 -std=c++17 "$CUDA_RESERVER_SRC" -o "$CUDA_RESERVER_BIN"
    fi
}

first_visible_gpu_for_nvidia_smi() {
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        printf "%s\n" "${CUDA_VISIBLE_DEVICES%%,*}"
    else
        printf "0\n"
    fi
}

compute_vllm_gpu_memory_utilization() {
    local memory_limit_gb=$1
    local nvidia_smi_gpu
    nvidia_smi_gpu=$(first_visible_gpu_for_nvidia_smi)
    local total_mib
    total_mib=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "$nvidia_smi_gpu" | head -n 1 | tr -d ' ')
    awk -v limit_gb="$memory_limit_gb" \
        -v total_mib="$total_mib" \
        -v headroom="$VLLM_GPU_MEMORY_HEADROOM" '
        BEGIN {
            total_gb = total_mib / 1024.0;
            util = (limit_gb / total_gb) * headroom;
            if (util > 0.99) {
                util = 0.99;
            }
            if (util <= 0 || util > 1) {
                exit 1;
            }
            printf "%.6f", util;
        }'
}

start_cuda_memory_reserver() {
    local name=$1
    local memory_limit_gb=$2
    local reserver_log="${LOG_DIR}/${name}.mem_reserver.log"
    local ready_file="${LOG_DIR}/${name}.mem_reserver.ready"

    build_cuda_memory_reserver
    cleanup_current_reserver
    rm -f "$ready_file"

    echo "  -> Reserving GPU memory outside ${memory_limit_gb} GiB limit"
    "$CUDA_RESERVER_BIN" \
        --device "$CUDA_RESERVER_DEVICE" \
        --leave-gb "$memory_limit_gb" \
        --ready-file "$ready_file" \
        > "$reserver_log" 2>&1 &
    CURRENT_RESERVER_PID=$!
    CURRENT_RESERVER_READY_FILE="$ready_file"
    printf "%s\n" "$CURRENT_RESERVER_PID" > "$CURRENT_RESERVER_PID_FILE"

    for _ in {1..120}; do
        if [[ -f "$ready_file" ]]; then
            return 0
        fi
        if ! kill -0 "$CURRENT_RESERVER_PID" 2>/dev/null; then
            echo "    [Error] CUDA memory reserver exited early. See $reserver_log"
            return 1
        fi
        sleep 0.5
    done

    echo "    [Error] CUDA memory reserver did not become ready. See $reserver_log"
    return 1
}

# Format: "ModelName:HFModelPath:NPP:NTG:NPL[:MemoryLimitGB]"
# SuperInfer num-prompts is NPL * NB to align with llama.cpp's NPL/NB setup.
runs=(
    "llama3.1-8b:/scratch/sliu27/models/HF/llama3.1-8b:1024:256:48"
    "llama3.1-8b:/scratch/sliu27/models/HF/llama3.1-8b:256:1024:48"
    "llama3.1-8b:/scratch/sliu27/models/HF/llama3.1-8b:512:512:60"
    "llama3.1-8b:/scratch/sliu27/models/HF/llama3.1-8b:1024:128:52"
    "llama3.1-8b:/scratch/sliu27/models/HF/llama3.1-8b:1024:64:56"
    "llama3.1-8b:/scratch/sliu27/models/HF/llama3.1-8b:1024:32:58"
)

run_superinfer_cmd() {
    local name=$1; local model_path=$2; local npp=$3; local ntg=$4; local npl=$5; local nb=$6; local memory_limit_gb=$7

    local num_prompts=$(( npl * nb ))
    local max_model_len=$(( npp + ntg ))
    local log_file="${LOG_DIR}/${name}.log"
    local json_file="${LOG_DIR}/${name}.json"
    local gpu_mem_utilization
    gpu_mem_utilization=$(compute_vllm_gpu_memory_utilization "$memory_limit_gb")

    start_cuda_memory_reserver "$name" "$memory_limit_gb"

    echo "  -> Executing: $name"
    echo "     GPU memory limit: ${memory_limit_gb} GiB; vLLM utilization: ${gpu_mem_utilization}"
    local full_cmd=(
        python benchmarks/benchmark_throughput.py
        --backend vllm
        --async-engine
        --model "$model_path"
        --num-prompts "$num_prompts"
        --input-len "$npp"
        --output-len "$ntg"
        --max-model-len "$max_model_len"
        --gpu-memory-utilization "$gpu_mem_utilization"
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
    cleanup_current_reserver

    if (( status != 0 )); then
        echo "    [Error] $name failed. See $log_file"
        return "$status"
    fi

    sleep 5
}

TOTAL=${#runs[@]}
COUNT=0

for entry in "${runs[@]}"; do
    IFS=':' read -r M_NAME HF_PATH NPP NTG NPL RUN_MEMORY_LIMIT_GB <<< "$entry"
    RUN_MEMORY_LIMIT_GB="${RUN_MEMORY_LIMIT_GB:-$GPU_MEMORY_LIMIT_GB}"

    COUNT=$((COUNT + 1))
    echo "=================================================="
    echo "[$COUNT/$TOTAL] Model: $M_NAME | NPP: $NPP | NTG: $NTG | NPL: $NPL | NB: $NB | MEM: ${RUN_MEMORY_LIMIT_GB}GiB"
    echo "=================================================="

    PREFIX="${M_NAME}_NPP${NPP}_NTG${NTG}_NPL${NPL}_NB${NB}"
    run_superinfer_cmd "${PREFIX}_SuperInfer" "$HF_PATH" "$NPP" "$NTG" "$NPL" "$NB" "$RUN_MEMORY_LIMIT_GB"

done

echo "=================================================="
echo "SuperInfer benchmarks completed. Logs saved in $LOG_DIR."
