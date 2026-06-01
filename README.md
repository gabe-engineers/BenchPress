# BenchPress

BenchPress is a lightweight benchmark harness for OpenAI-compatible text
completion endpoints. It generates prompts, streams completions, and renders a
live terminal dashboard with throughput, TTFT, recent successes, and recent
errors.

## Requirements

- Python 3.12+
- `uv`
- A reachable OpenAI-compatible endpoint
- An API key for that endpoint

## Setup

```bash
uv sync
```

## Required environment variables

For `--provider vllm`:

- `BENCHPRESS_API_KEY`: bearer token for the target endpoint
- `VLLM_BASE_URL`: vLLM server URL, either the server root or a `/v1` URL

Example:

```bash
export BENCHPRESS_API_KEY='your_api_key'
export VLLM_BASE_URL='https://your-endpoint.example.com'
```

`benchpress.py` normalizes `VLLM_BASE_URL` to the OpenAI-compatible `/v1` base
URL automatically, so both of these work:

```bash
export VLLM_BASE_URL='https://your-endpoint.example.com'
export VLLM_BASE_URL='https://your-endpoint.example.com/v1'
```

For `--provider openai`, `VLLM_BASE_URL` is not required.

## Basic usage

Run with the defaults:

```bash
make run
```

Equivalent direct invocation:

```bash
.venv/bin/python benchpress.py
```

Typical vLLM example:

```bash
export BENCHPRESS_API_KEY='your_api_key'
export VLLM_BASE_URL='https://your-endpoint.example.com'

.venv/bin/python benchpress.py \
  --provider vllm \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --traffic-type constant \
  --input-size-tokens 100
```

Runpod example:

```bash
export BENCHPRESS_API_KEY='your_api_key'
export VLLM_BASE_URL='https://bt7omfs8p50vdy-8000.proxy.runpod.net'

make run
```

## CLI options

- `--provider {vllm,openai}`: target backend, default `vllm`
- `--model`: model name to request, default
  `meta-llama/Llama-3.1-8B-Instruct`
- `--traffic-type {constant,bursty}`: request pattern, default `constant`
- `--input-size-tokens`: generated prompt size in tokens, default `100`

Traffic modes:

- `constant`: sends 1 request, then waits 1 second
- `bursty`: sends 10 concurrent requests, then waits 10 seconds

## Runtime controls

While `benchpress.py` is running in an interactive terminal:

- Press `p` to pause or resume traffic generation.
- Press `q` to stop starting new requests, wait for in-flight requests to
  finish, and exit cleanly.

Behavior notes:

- Pausing does not cancel in-flight requests.
- Displayed runtime and throughput exclude time spent paused.
- The dashboard shows recent success and error logs and adapts to terminal
  width.

## Build a binary

```bash
make build
```

This creates a standalone binary in `dist/benchpress`.

## Notes

- The harness currently uses the OpenAI-compatible `/v1/completions` API, not
  `/v1/chat/completions`.
- If you export `BENCHMARK_API_KEY`, the harness will not see it. The expected
  variable name is `BENCHPRESS_API_KEY`.
- The Dockerfile, compose file, and publish script in this repo are optional
  helpers for running a local vLLM server or packaging an image. They are not
  required to use the benchmark harness itself.
