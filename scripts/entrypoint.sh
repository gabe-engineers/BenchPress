#!/bin/sh

set -eu

MODEL_ID="${MODEL_ID:-meta-llama/Llama-3.1-8B-Instruct}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-10240}"

export HF_TOKEN="${HF_TOKEN:-${HF_ACCESS_TOKEN:-}}"

has_max_model_len=0
for arg in "$@"; do
    case "$arg" in
        --max-model-len|--max-model-len=*)
            has_max_model_len=1
            break
            ;;
    esac
done

if [ "$has_max_model_len" -eq 0 ]; then
    set -- --max-model-len "$MAX_MODEL_LEN" "$@"
fi

if [ -n "${VLLM_API_KEY:-}" ]; then
    exec vllm serve "$MODEL_ID" --host "$HOST" --port "$PORT" --api-key "$VLLM_API_KEY" "$@"
fi

exec vllm serve "$MODEL_ID" --host "$HOST" --port "$PORT" "$@"
