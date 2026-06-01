# vLLM Llama 3.1 server image

This directory builds a local Docker image that starts a vLLM OpenAI-compatible
server for `meta-llama/Llama-3.1-8B-Instruct`.

`Llama 3.1 7B` is not an actual Meta model name. The closest matching model is
`meta-llama/Llama-3.1-8B-Instruct`, which is the default here.

## Requirements

- Docker with Compose support
- An NVIDIA GPU that can run the official CUDA vLLM image
- A Hugging Face token with access to the gated Llama 3.1 model

## Build

```bash
docker build -t vllm-llama31-8b:local .
```

## Run with docker

```bash
export HF_TOKEN=your_hugging_face_token
export VLLM_API_KEY=your_server_api_key

docker run --rm \
  --gpus all \
  --ipc=host \
  -p 8000:8000 \
  -e HF_TOKEN \
  -e MAX_MODEL_LEN=10240 \
  -e VLLM_API_KEY \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  vllm-llama31-8b:local
```

## Run with compose

```bash
cp .env.example .env
# Fill in HF_TOKEN and optionally VLLM_API_KEY in .env.
docker compose up --build
```

For Runpod, `HF_ACCESS_TOKEN` is also accepted. The container maps
`HF_ACCESS_TOKEN` to `HF_TOKEN` at startup.
The image defaults `MAX_MODEL_LEN` to `10240` on startup unless you pass an
explicit `--max-model-len` argument.

## Test

```bash
curl \
  -H "Authorization: Bearer your_server_api_key" \
  http://localhost:8000/v1/models
```

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your_server_api_key" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "messages": [
      {"role": "user", "content": "Say hello in one sentence."}
    ],
    "max_completion_tokens": 64
  }'
```

## Notes

- Pass additional vLLM server args after the image name, for example
  `--tensor-parallel-size 2`.
- Change the served model by setting `MODEL_ID`.
- Set `VLLM_BASE_URL` to your server root or `/v1`; `benchpress.py` normalizes
  it to the OpenAI-compatible `/v1` base URL automatically.
- Change the startup context limit by setting `MAX_MODEL_LEN`.
- By default, compose stores the Hugging Face cache in `./.hf-cache`.
- If `VLLM_API_KEY` is set, the server requires `Authorization: Bearer ...`.
- For gated Hugging Face models, set either `HF_TOKEN` or `HF_ACCESS_TOKEN`.

## Runtime controls

When `benchpress.py` is running in an interactive terminal:

- Press `p` to pause or resume traffic generation.
- Press `q` to stop starting new requests, wait for in-flight requests to
  finish, and exit cleanly.
- Pausing does not cancel in-flight requests. It only stops new requests from
  being started until you resume.
- The displayed runtime and throughput rates exclude time spent paused.
- The dashboard shows recent success and error logs, adapting to terminal width.

## Publish to Docker Hub

Start Docker first, then build and push:

```bash
docker login
chmod +x scripts/publish.sh
IMAGE_TAG=latest ./scripts/publish.sh
```

By default, `scripts/publish.sh` publishes `linux/amd64`, which is the safe
default for most NVIDIA GPU hosts. For that single-platform case, the script
now builds locally and then runs a plain `docker push`, which is often more
reliable on Docker Hub than `buildx --push`.

Override `PLATFORMS` if you need something else. Multi-platform publishes still
use `buildx --push`, for example:

```bash
PLATFORMS=linux/amd64,linux/arm64 IMAGE_TAG=latest ./scripts/publish.sh
```
