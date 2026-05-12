#!/usr/bin/env bash
set -euo pipefail

has_arg() {
    local needle="$1"
    shift
    local arg
    for arg in "$@"; do
        if [ "${arg}" = "${needle}" ] || [[ "${arg}" == "${needle}="* ]]; then
            return 0
        fi
    done
    return 1
}

if [ "$#" -gt 0 ]; then
    args=("$@")

    if ! has_arg "--host" "${args[@]}"; then
        args+=("--host" "${DAMORK_HOST}")
    fi
    if ! has_arg "--port" "${args[@]}"; then
        args+=("--port" "${DAMORK_PORT}")
    fi
    if ! has_arg "--served-model-name" "${args[@]}"; then
        args+=("--served-model-name" "${DAMORK_SERVED_MODEL_NAME}")
    fi
    if ! has_arg "--max-model-len" "${args[@]}"; then
        args+=("--max-model-len" "${DAMORK_MAX_MODEL_LEN}")
    fi
    if ! has_arg "--gpu-memory-utilization" "${args[@]}"; then
        args+=("--gpu-memory-utilization" "${DAMORK_GPU_MEMORY_UTILIZATION}")
    fi
    if ! has_arg "--enforce-eager" "${args[@]}"; then
        args+=("--enforce-eager")
    fi

    exec vllm serve "${args[@]}"
fi

exec vllm serve "${DAMORK_MODEL}" \
    --host "${DAMORK_HOST}" \
    --port "${DAMORK_PORT}" \
    --served-model-name "${DAMORK_SERVED_MODEL_NAME}" \
    --max-model-len "${DAMORK_MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${DAMORK_GPU_MEMORY_UTILIZATION}" \
    --enforce-eager
