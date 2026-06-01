ARG VLLM_IMAGE=vllm/vllm-openai:latest
FROM ${VLLM_IMAGE}

ENV MODEL_ID=meta-llama/Llama-3.1-8B-Instruct \
    HOST=0.0.0.0 \
    PORT=8000 \
    MAX_MODEL_LEN=10240

COPY scripts/entrypoint.sh /usr/local/bin/benchpress-entrypoint
RUN chmod +x /usr/local/bin/benchpress-entrypoint

EXPOSE 8000

ENTRYPOINT ["/usr/local/bin/benchpress-entrypoint"]
CMD []
