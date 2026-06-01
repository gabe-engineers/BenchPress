ARG VLLM_IMAGE=vllm/vllm-openai:latest
FROM ${VLLM_IMAGE}

ENV MODEL_ID=meta-llama/Llama-3.1-8B-Instruct \
    HOST=0.0.0.0 \
    PORT=8000

EXPOSE 8000

ENTRYPOINT ["/bin/sh", "-lc", "export HF_TOKEN=\"${HF_TOKEN:-${HF_ACCESS_TOKEN:-}}\"; if [ -n \"${VLLM_API_KEY:-}\" ]; then exec vllm serve \"${MODEL_ID:-meta-llama/Llama-3.1-8B-Instruct}\" --host \"${HOST:-0.0.0.0}\" --port \"${PORT:-8000}\" --api-key \"${VLLM_API_KEY}\" \"$@\"; fi; exec vllm serve \"${MODEL_ID:-meta-llama/Llama-3.1-8B-Instruct}\" --host \"${HOST:-0.0.0.0}\" --port \"${PORT:-8000}\" \"$@\"", "--"]
CMD []
