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
  --traffic-volume 5 \
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
- `--traffic-volume`: requests to send each traffic cadence; defaults to `1`
  for `constant` and `10` for `bursty`
- `--input-size-tokens`: generated prompt size in tokens, default `100`
- `--run <name>`: execute a 60-second named run and write per-sample metrics to
  `runs/<name>.csv`

Traffic modes:

- `constant`: sends `traffic_volume` concurrent requests, then waits 1 second
- `bursty`: sends `traffic_volume` concurrent requests, then waits 10 seconds

Run capture example:

```bash
.venv/bin/python benchpress.py \
  --traffic-type constant \
  --traffic-volume 3 \
  --input-size-tokens 100 \
  --run baseline-100toks
```

This writes a seaborn-friendly CSV to `runs/baseline-100toks.csv` with one row
per sampled second. Each row includes the run name, sample timestamp,
`sample_second`, `tokens_per_second`, `avg_ttft_seconds`,
`avg_request_latency_seconds`, request/error counts, and the other core
dashboard metrics.

## Experiments

Run a multi-step experiment from YAML:

```bash
uv run benchpress-experiment examples/cache-toggle.experiment.yaml
```

Equivalent direct invocation:

```bash
.venv/bin/python benchpress.py experiment examples/cache-toggle.experiment.yaml
```

The YAML file defines named runs, optional shared defaults, and the final
comparison output. BenchPress executes the runs sequentially, writes one CSV per
run, and compares only the CSVs produced by that experiment so older files in
`runs/` do not leak into the chart.

Example:

```yaml
name: cache-toggle
runs_dir: runs/cache-toggle

defaults:
  provider: vllm
  model: meta-llama/Llama-3.1-8B-Instruct
  traffic_type: constant
  traffic_volume: 1
  input_size_tokens: 100

comparison:
  title: Cache Toggle Comparison
  output: runs/cache-toggle/cache-toggle-comparison.png

runs:
  - name: cache-off
    prompt_after_run: Enable caching, then press any key to start the next run.
  - name: cache-on
```

Supported experiment fields:

- `name`: optional experiment name, used in the default comparison filename
- `runs_dir`: optional output directory for the run CSV files
- `defaults`: optional shared run settings for `provider`, `model`,
  `traffic_type`, `traffic_volume`, and `input_size_tokens`
- `prompt_between_runs`: optional global pause between runs; set it to `true`
  for the default prompt, `false` to disable, or a custom string prompt
- `comparison.enabled`: optional boolean, defaults to `true`
- `comparison.metrics`: optional metric list, defaults to the same metrics used
  by `compare-runs`
- `comparison.output`: optional output image path
- `comparison.title`: optional chart title
- `runs`: required list of runs; each run needs a unique `name` and can override
  `provider`, `model`, `traffic_type`, `traffic_volume`, `input_size_tokens`, and
  `prompt_after_run`

`prompt_after_run` uses the same values as `prompt_between_runs`: `true`,
`false`, or a custom string. Relative paths in the YAML are resolved from the
YAML file's directory.

Compare saved runs:

```bash
uv run compare-runs
```

Equivalent direct invocation:

```bash
.venv/bin/python benchpress.py compare-runs
```

By default this reads every `*.csv` in `runs/` and writes
`runs/compare-runs.png`. The figure uses seaborn to plot one chart per metric,
overlaying all selected runs in each chart with distinct colors and a shared
legend.

You can narrow the plotted metrics or change the output path:

```bash
uv run compare-runs \
  --metrics tokens_per_second avg_ttft_seconds avg_request_latency_seconds \
  --output runs/latency-throughput-comparison.png
```

Minimal ad hoc plotting example:

```python
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

df = pd.read_csv("runs/baseline-100toks.csv")
sns.lineplot(data=df, x="sample_second", y="tokens_per_second")
plt.show()
```

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
