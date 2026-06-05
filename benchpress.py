import argparse
import asyncio
import atexit
import csv
import re
import shutil
import select
import sys
import termios
import textwrap
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from os import environ, system
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Any, List, Sequence
from urllib.parse import urlsplit, urlunsplit

from faker import Faker
from openai import AsyncOpenAI

from contracts import Provider
from tokenizer import Tokenizer


DEFAULT_COMPARE_RUN_METRICS = [
    "tokens_per_second",
    "avg_ttft_seconds",
    "avg_request_latency_seconds",
    "error_rate_percent",
]
DEFAULT_TRAFFIC_TYPE = "constant"
DEFAULT_TRAFFIC_VOLUMES = {
    "constant": 1,
    "bursty": 10,
}
TRAFFIC_CADENCE_SECONDS = {
    "constant": 1,
    "bursty": 10,
}
DEFAULT_INPUT_SIZE_TOKENS = 100
DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_PROVIDER = Provider.VLLM.value
DEFAULT_RUNS_DIR = Path("runs")
DEFAULT_EXPERIMENT_PAUSE_PROMPT = (
    "Run complete. Press any key to start the next run."
)


def sanitize_run_name(run_name: str) -> str:
    safe_run_name = re.sub(r"[^A-Za-z0-9._-]+", "_", run_name).strip("._")
    return safe_run_name or "run"


def positive_int_arg(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def default_traffic_volume(traffic_type: str) -> int:
    try:
        return DEFAULT_TRAFFIC_VOLUMES[traffic_type]
    except KeyError as exc:
        raise ValueError(f"Unsupported traffic type: {traffic_type}") from exc


def resolve_traffic_volume(
    value: object | None, *, traffic_type: str, field_name: str
) -> int:
    if value is None:
        return default_traffic_volume(traffic_type)
    return parse_positive_int(value, field_name=field_name)


def format_traffic_profile(traffic_type: str, traffic_volume: int) -> str:
    try:
        cadence_seconds = TRAFFIC_CADENCE_SECONDS[traffic_type]
    except KeyError as exc:
        raise ValueError(f"Unsupported traffic type: {traffic_type}") from exc
    return f"{traffic_volume} req/{cadence_seconds}s"


def build_benchmark_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="BenchPress a light weight LLM endpoint benchmark harness"
    )
    parser.add_argument(
        "--traffic-type",
        type=str,
        choices=["constant", "bursty"],
        default=DEFAULT_TRAFFIC_TYPE,
    )
    parser.add_argument(
        "--traffic-volume",
        type=positive_int_arg,
        default=None,
        help=(
            "Requests to send each traffic cadence. Defaults to 1 for constant "
            "and 10 for bursty."
        ),
    )
    parser.add_argument("--input-size-tokens", type=int, default=DEFAULT_INPUT_SIZE_TOKENS)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument(
        "--provider", type=str, choices=["vllm", "openai"], default=DEFAULT_PROVIDER
    )
    parser.add_argument(
        "--run",
        type=str,
        help=(
            "Capture a 60-second metrics run and write time-series data to "
            "runs/<name>.csv"
        ),
    )
    return parser


def build_experiment_parser(prog: str = "experiment") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description=(
            "Run a YAML-defined sequence of named benchmark runs and compare them."
        ),
    )
    parser.add_argument(
        "config",
        type=Path,
        help="Path to an experiment YAML config file.",
    )
    return parser


def build_compare_runs_parser(prog: str = "compare-runs") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description=(
            "Generate a seaborn time-series comparison from the CSV files in runs/."
        ),
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("runs"),
        help="Directory containing run CSV files. Default: runs",
    )
    parser.add_argument(
        "--glob",
        type=str,
        default="*.csv",
        help="Glob for selecting run CSV files inside --runs-dir. Default: *.csv",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=list(DEFAULT_COMPARE_RUN_METRICS),
        help=(
            "Metric columns to plot. Default: "
            + ", ".join(DEFAULT_COMPARE_RUN_METRICS)
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs") / "compare-runs.png",
        help="Output image path. Default: runs/compare-runs.png",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="BenchPress Run Comparison",
        help="Figure title.",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "experiment":
        args = build_experiment_parser(prog="benchpress.py experiment").parse_args(
            argv[1:]
        )
        args.command = "experiment"
        return args
    if argv and argv[0] == "compare-runs":
        args = build_compare_runs_parser(prog="benchpress.py compare-runs").parse_args(
            argv[1:]
        )
        args.command = "compare-runs"
        return args

    args = build_benchmark_parser().parse_args(argv)
    args.command = "benchmark"
    return args


def load_compare_runs_dependencies():
    cache_root = Path("/tmp") / "benchpress-plot-cache"
    matplotlib_cache_dir = cache_root / "matplotlib"
    xdg_cache_dir = cache_root / "xdg-cache"
    matplotlib_cache_dir.mkdir(parents=True, exist_ok=True)
    xdg_cache_dir.mkdir(parents=True, exist_ok=True)
    environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache_dir))
    environ.setdefault("XDG_CACHE_HOME", str(xdg_cache_dir))

    try:
        import matplotlib
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "compare-runs requires pandas, seaborn, and matplotlib. Run `uv sync` "
            "after pulling the updated dependencies."
        ) from exc

    matplotlib.use("Agg")

    try:
        import pandas as pd
        import seaborn as sns
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "compare-runs requires pandas, seaborn, and matplotlib. Run `uv sync` "
            "after pulling the updated dependencies."
        ) from exc

    from matplotlib import pyplot as plt

    return pd, sns, plt


def load_experiment_dependencies():
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "experiment requires PyYAML. Run `uv sync` after pulling the updated "
            "dependencies."
        ) from exc

    return yaml


@dataclass(frozen=True)
class ExperimentRun:
    name: str
    provider: Provider
    model: str
    traffic_type: str
    traffic_volume: int
    input_size_tokens: int
    prompt_after_run: str | None


@dataclass(frozen=True)
class ExperimentComparison:
    enabled: bool
    metrics: List[str]
    output_path: Path
    title: str


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    config_path: Path
    runs_dir: Path
    runs: List[ExperimentRun]
    comparison: ExperimentComparison


@dataclass(frozen=True)
class ExperimentResult:
    run_output_paths: List[Path]
    comparison_output_path: Path | None


def parse_prompt_setting(
    value: object, *, field_name: str, default_prompt: str
) -> str | None:
    if value is None or value is False:
        return None
    if value is True:
        return default_prompt
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            raise ValueError(f"{field_name} must not be blank when set as a string")
        return stripped
    raise ValueError(f"{field_name} must be true, false, or a non-empty string")


def resolve_config_relative_path(config_path: Path, raw_path: str | None, default: Path) -> Path:
    path = default if raw_path is None else Path(raw_path)
    if path.is_absolute():
        return path
    return (config_path.parent / path).resolve()


def parse_positive_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def parse_required_string(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def parse_experiment_mapping(value: object, *, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be a mapping")
    return value


def parse_experiment_run(
    run_data: object,
    *,
    run_index: int,
    defaults: dict[str, Any],
    inherited_prompt: str | None,
) -> ExperimentRun:
    if not isinstance(run_data, dict):
        raise ValueError(f"runs[{run_index}] must be a mapping")

    name = parse_required_string(run_data.get("name"), field_name=f"runs[{run_index}].name")
    provider_raw = run_data.get("provider", defaults["provider"])
    try:
        provider = Provider(parse_required_string(provider_raw, field_name=f"runs[{run_index}].provider"))
    except ValueError as exc:
        raise ValueError(
            f"runs[{run_index}].provider must be one of: "
            f"{', '.join(provider.value for provider in Provider)}"
        ) from exc

    model = parse_required_string(
        run_data.get("model", defaults["model"]),
        field_name=f"runs[{run_index}].model",
    )
    traffic_type = parse_required_string(
        run_data.get("traffic_type", defaults["traffic_type"]),
        field_name=f"runs[{run_index}].traffic_type",
    )
    if traffic_type not in {"constant", "bursty"}:
        raise ValueError(
            f"runs[{run_index}].traffic_type must be one of: constant, bursty"
        )

    traffic_volume = resolve_traffic_volume(
        run_data.get("traffic_volume", defaults.get("traffic_volume")),
        traffic_type=traffic_type,
        field_name=f"runs[{run_index}].traffic_volume",
    )

    input_size_tokens = parse_positive_int(
        run_data.get("input_size_tokens", defaults["input_size_tokens"]),
        field_name=f"runs[{run_index}].input_size_tokens",
    )

    if "prompt_after_run" in run_data:
        prompt_after_run = parse_prompt_setting(
            run_data["prompt_after_run"],
            field_name=f"runs[{run_index}].prompt_after_run",
            default_prompt=DEFAULT_EXPERIMENT_PAUSE_PROMPT,
        )
    else:
        prompt_after_run = inherited_prompt

    return ExperimentRun(
        name=name,
        provider=provider,
        model=model,
        traffic_type=traffic_type,
        traffic_volume=traffic_volume,
        input_size_tokens=input_size_tokens,
        prompt_after_run=prompt_after_run,
    )


def load_experiment_config(config_path: Path) -> ExperimentConfig:
    yaml = load_experiment_dependencies()
    with config_path.open() as config_file:
        raw_config = yaml.safe_load(config_file) or {}

    if not isinstance(raw_config, dict):
        raise ValueError("Experiment config root must be a mapping")

    config_path = config_path.resolve()
    defaults = parse_experiment_mapping(
        raw_config.get("defaults"), field_name="defaults"
    )
    comparison_data = parse_experiment_mapping(
        raw_config.get("comparison"), field_name="comparison"
    )
    experiment_name = raw_config.get("name", config_path.stem)
    if not isinstance(experiment_name, str) or not experiment_name.strip():
        raise ValueError("name must be a non-empty string when provided")
    experiment_name = experiment_name.strip()

    runs_dir_raw = raw_config.get("runs_dir")
    if runs_dir_raw is not None and not isinstance(runs_dir_raw, str):
        raise ValueError("runs_dir must be a string path")
    runs_dir = resolve_config_relative_path(
        config_path, runs_dir_raw, DEFAULT_RUNS_DIR
    )

    inherited_prompt = parse_prompt_setting(
        raw_config.get("prompt_between_runs"),
        field_name="prompt_between_runs",
        default_prompt=DEFAULT_EXPERIMENT_PAUSE_PROMPT,
    )

    merged_defaults = {
        "provider": defaults.get("provider", DEFAULT_PROVIDER),
        "model": defaults.get("model", DEFAULT_MODEL),
        "traffic_type": defaults.get("traffic_type", DEFAULT_TRAFFIC_TYPE),
        "traffic_volume": defaults.get("traffic_volume"),
        "input_size_tokens": defaults.get(
            "input_size_tokens", DEFAULT_INPUT_SIZE_TOKENS
        ),
    }

    runs_data = raw_config.get("runs")
    if not isinstance(runs_data, list) or not runs_data:
        raise ValueError("runs must be a non-empty list")

    runs = [
        parse_experiment_run(
            run_data,
            run_index=index,
            defaults=merged_defaults,
            inherited_prompt=inherited_prompt,
        )
        for index, run_data in enumerate(runs_data)
    ]

    seen_run_output_paths: set[Path] = set()
    for run in runs:
        run_output_path = runs_dir / f"{sanitize_run_name(run.name)}.csv"
        if run_output_path in seen_run_output_paths:
            raise ValueError(
                "Experiment run names must be unique after filename sanitization"
            )
        seen_run_output_paths.add(run_output_path)

    comparison_enabled = comparison_data.get("enabled", True)
    if not isinstance(comparison_enabled, bool):
        raise ValueError("comparison.enabled must be a boolean")

    comparison_metrics = comparison_data.get(
        "metrics", list(DEFAULT_COMPARE_RUN_METRICS)
    )
    if not isinstance(comparison_metrics, list) or not comparison_metrics:
        raise ValueError("comparison.metrics must be a non-empty list when provided")
    if not all(isinstance(metric, str) and metric.strip() for metric in comparison_metrics):
        raise ValueError("comparison.metrics must contain non-empty strings")

    comparison_output_raw = comparison_data.get("output")
    if comparison_output_raw is not None and not isinstance(comparison_output_raw, str):
        raise ValueError("comparison.output must be a string path")
    comparison_output_path = resolve_config_relative_path(
        config_path,
        comparison_output_raw,
        runs_dir / f"{sanitize_run_name(experiment_name)}-comparison.png",
    )
    comparison_title = comparison_data.get(
        "title", f"BenchPress Experiment Comparison: {experiment_name}"
    )
    if not isinstance(comparison_title, str) or not comparison_title.strip():
        raise ValueError("comparison.title must be a non-empty string when provided")

    return ExperimentConfig(
        name=experiment_name,
        config_path=config_path,
        runs_dir=runs_dir,
        runs=runs,
        comparison=ExperimentComparison(
            enabled=comparison_enabled,
            metrics=[metric.strip() for metric in comparison_metrics],
            output_path=comparison_output_path,
            title=comparison_title.strip(),
        ),
    )


def wait_for_keypress(prompt: str):
    message = f"{prompt.strip()} "
    if not sys.stdin.isatty():
        print(f"{message}(skipping prompt because stdin is not interactive)")
        return

    fd = None
    original_terminal_settings = None
    print(message, end="", flush=True)
    try:
        fd = sys.stdin.fileno()
        original_terminal_settings = termios.tcgetattr(fd)
        updated_terminal_settings = termios.tcgetattr(fd)
        updated_terminal_settings[3] &= ~(termios.ICANON | termios.ECHO)
        updated_terminal_settings[6][termios.VMIN] = 1
        updated_terminal_settings[6][termios.VTIME] = 0
        termios.tcsetattr(fd, termios.TCSANOW, updated_terminal_settings)
        sys.stdin.read(1)
    except (OSError, termios.error):
        print()
        input("Press Enter to continue...")
    finally:
        if fd is not None and original_terminal_settings is not None:
            try:
                termios.tcsetattr(fd, termios.TCSANOW, original_terminal_settings)
            except (OSError, termios.error):
                pass
        print()


def compare_run_files(
    run_files: Sequence[Path],
    metrics: Sequence[str],
    output_path: Path,
    title: str,
) -> Path:
    pd, sns, plt = load_compare_runs_dependencies()
    metrics = list(metrics)
    run_files = [Path(run_file) for run_file in run_files]
    if not run_files:
        raise FileNotFoundError("No run CSV files were provided for comparison")

    missing_run_files = [run_file for run_file in run_files if not run_file.exists()]
    if missing_run_files:
        missing_paths = ", ".join(str(run_file) for run_file in missing_run_files)
        raise FileNotFoundError(f"Run CSV files were not found: {missing_paths}")

    frames = []
    missing_columns: dict[str, list[str]] = {}
    run_order: List[str] = []
    for run_file in run_files:
        frame = pd.read_csv(run_file)
        if "sample_second" not in frame.columns:
            raise ValueError(f"{run_file} is missing required column 'sample_second'")

        if "run_name" not in frame.columns:
            frame["run_name"] = run_file.stem

        missing = [metric for metric in metrics if metric not in frame.columns]
        if missing:
            missing_columns[run_file.name] = missing
            continue

        frame = frame[["run_name", "sample_second", *metrics]].copy()
        frame["run_name"] = frame["run_name"].fillna(run_file.stem).astype(str)
        run_order.extend(frame["run_name"].drop_duplicates().tolist())
        frames.append(frame)

    if missing_columns:
        missing_details = ", ".join(
            f"{file_name}: {', '.join(columns)}"
            for file_name, columns in missing_columns.items()
        )
        raise ValueError(f"Missing metric columns in run CSVs: {missing_details}")

    if not frames:
        raise ValueError("No run CSVs contained the requested metrics")

    combined = pd.concat(frames, ignore_index=True)
    combined["run_name"] = pd.Categorical(
        combined["run_name"],
        categories=list(dict.fromkeys(run_order)),
        ordered=True,
    )
    run_names = list(combined["run_name"].cat.categories)

    sns.set_theme(style="whitegrid", context="notebook")
    figure, axes = plt.subplots(
        nrows=len(metrics),
        ncols=1,
        figsize=(7.2, 2.8 * len(metrics)),
        squeeze=False,
        sharex=True,
    )
    palette = dict(zip(run_names, sns.color_palette("deep", n_colors=len(run_names))))

    for row_index, metric in enumerate(metrics):
        axis = axes[row_index][0]
        sns.lineplot(
            data=combined,
            x="sample_second",
            y=metric,
            hue="run_name",
            hue_order=run_names,
            palette=palette,
            ax=axis,
            linewidth=2,
            legend=False,
        )
        axis.set_xlim(left=0)
        axis.set_ylabel(metric)
        axis.set_xlabel("Sample second" if row_index == len(metrics) - 1 else "")
        axis.set_title(metric)

    from matplotlib.lines import Line2D

    figure.legend(
        handles=[
            Line2D([0], [0], color=palette[run_name], linewidth=2, label=run_name)
            for run_name in run_names
        ],
        title="Run",
        loc="upper center",
        ncol=min(len(run_names), 4),
        bbox_to_anchor=(0.5, 1.02),
    )
    figure.suptitle(title, y=1.06)
    figure.tight_layout(rect=(0, 0, 1, 0.97))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return output_path


def compare_runs(
    runs_dir: Path,
    glob_pattern: str,
    metrics: Sequence[str],
    output_path: Path,
    title: str,
) -> Path:
    run_files = sorted(runs_dir.glob(glob_pattern))
    if not run_files:
        raise FileNotFoundError(
            f"No run CSV files matched {glob_pattern!r} in {runs_dir}"
        )
    return compare_run_files(
        run_files=run_files,
        metrics=metrics,
        output_path=output_path,
        title=title,
    )


def compare_runs_cli(argv: Sequence[str] | None = None) -> int:
    args = build_compare_runs_parser().parse_args(
        list(argv) if argv is not None else None
    )
    output_path = compare_runs(
        runs_dir=args.runs_dir,
        glob_pattern=args.glob,
        metrics=args.metrics,
        output_path=args.output,
        title=args.title,
    )
    print(f"Wrote run comparison plot to {output_path}")
    return 0


@dataclass
class Metrics:
    started_at: float
    inflight: int
    total_requests: int
    total_success: int
    total_errors: int
    lock: Lock
    total_tokens: int
    total_paused_duration: float = 0.0
    paused_at: float | None = None
    ttfts: List[float] = field(default_factory=list)
    request_latencies: List[float] = field(default_factory=list)
    recent_successes: List[str] = field(default_factory=list)
    recent_errors: List[str] = field(default_factory=list)

    def record_request(self):
        with self.lock:
            self.inflight += 1
            self.total_requests += 1

    def record_success(self, success_message: str):
        with self.lock:
            self.inflight -= 1
            self.total_success += 1
            self.recent_successes.append(success_message)
            self.recent_successes = self.recent_successes[-6:]

    def record_failure(self, error_message: str):
        with self.lock:
            self.inflight -= 1
            self.total_errors += 1
            self.recent_errors.append(error_message)
            self.recent_errors = self.recent_errors[-6:]

    def increment_total_tokens(self, tokens: int):
        with self.lock:
            self.total_tokens += tokens

    def add_ttft(self, ttft: float):
        with self.lock:
            self.ttfts.append(ttft)

    def record_request_latency(self, latency: float):
        with self.lock:
            self.request_latencies.append(latency)


@dataclass
class MetricsSnapshot:
    captured_at: str
    runtime_seconds: float
    inflight: int
    total_requests: int
    total_success: int
    total_errors: int
    total_tokens: int
    tokens_per_second: float
    avg_ttft_seconds: float
    avg_request_latency_seconds: float
    error_rate_percent: float
    status: str
    recent_successes: List[str]
    recent_errors: List[str]

    def to_timeseries_point(self, sample_second: float) -> dict:
        return {
            "sample_second": sample_second,
            "captured_at": self.captured_at,
            "runtime_seconds": self.runtime_seconds,
            "inflight": self.inflight,
            "total_requests": self.total_requests,
            "total_success": self.total_success,
            "total_errors": self.total_errors,
            "total_tokens": self.total_tokens,
            "tokens_per_second": self.tokens_per_second,
            "avg_ttft_seconds": self.avg_ttft_seconds,
            "avg_request_latency_seconds": self.avg_request_latency_seconds,
            "error_rate_percent": self.error_rate_percent,
            "status": self.status,
        }


class RequestGenerator:
    RUN_DURATION_SECONDS = 60
    RUN_SAMPLE_INTERVAL_SECONDS = 1

    def __init__(
        self,
        model: str,
        provider: Provider,
        run_name: str | None = None,
        run_output_dir: Path = DEFAULT_RUNS_DIR,
    ):
        api_key, base_url = environ.get("BENCHPRESS_API_KEY"), environ.get("VLLM_BASE_URL")
        if not api_key:
            raise Exception("You must set BENCHPRESS_API_KEY")
        if provider == Provider.VLLM and not base_url:
            raise Exception("You must set VLLM_BASE_URL")
        if provider == Provider.VLLM:
            base_url = self.normalize_vllm_base_url(base_url)
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.faker = Faker()
        self.model = model
        self.provider = provider
        self.tokenizer = Tokenizer(model, provider)
        self.lock = Lock()
        self.run_event = Event()
        self.run_event.set()
        self.shutdown_requested = False
        self.shutdown_reason: str | None = None
        self.controls_enabled = False
        self.stdin_fd = None
        self.original_terminal_settings = None
        self.run_name = run_name
        self.run_output_dir = run_output_dir
        self.traffic_type: str | None = None
        self.traffic_volume: int | None = None
        self.input_size_tokens: int | None = None
        self.run_started_at = datetime.now(timezone.utc).isoformat()
        self.run_samples: List[dict] = []
        self.run_output_path = (
            self.build_run_output_path(run_name, run_output_dir) if run_name else None
        )
        self.run_capture_thread: Thread | None = None
        self.run_file_written = False
        self.render_stop_event = Event()
        self.render_thread = Thread(target=self.render_dashboard, daemon=True)
        self.metrics = Metrics(
            started_at=time.perf_counter(),
            inflight=0,
            total_requests=0,
            total_success=0,
            total_errors=0,
            lock=self.lock,
            total_tokens=0,
        )
        self.enable_controls()
        self.render_thread.start()

    @staticmethod
    def build_run_output_path(run_name: str, run_output_dir: Path = DEFAULT_RUNS_DIR) -> Path:
        return run_output_dir / f"{sanitize_run_name(run_name)}.csv"

    @staticmethod
    def normalize_vllm_base_url(base_url: str) -> str:
        parsed_url = urlsplit(base_url)
        normalized_path = parsed_url.path.rstrip("/")
        if not normalized_path.endswith("/v1"):
            normalized_path = f"{normalized_path}/v1" if normalized_path else "/v1"
        return urlunsplit(
            (
                parsed_url.scheme,
                parsed_url.netloc,
                normalized_path,
                parsed_url.query,
                parsed_url.fragment,
            )
        )

    def enable_controls(self):
        if not sys.stdin.isatty():
            return

        try:
            self.stdin_fd = sys.stdin.fileno()
            self.original_terminal_settings = termios.tcgetattr(self.stdin_fd)
            updated_terminal_settings = termios.tcgetattr(self.stdin_fd)
            updated_terminal_settings[3] &= ~(termios.ICANON | termios.ECHO)
            updated_terminal_settings[6][termios.VMIN] = 1
            updated_terminal_settings[6][termios.VTIME] = 0
            termios.tcsetattr(
                self.stdin_fd, termios.TCSANOW, updated_terminal_settings
            )
        except (OSError, termios.error):
            self.stdin_fd = None
            self.original_terminal_settings = None
            return

        self.controls_enabled = True
        atexit.register(self.restore_terminal)
        Thread(target=self.listen_for_controls, daemon=True).start()

    def restore_terminal(self):
        if self.stdin_fd is None or self.original_terminal_settings is None:
            return

        try:
            termios.tcsetattr(
                self.stdin_fd, termios.TCSANOW, self.original_terminal_settings
            )
        except (OSError, termios.error):
            pass
        finally:
            self.stdin_fd = None
            self.original_terminal_settings = None

    def listen_for_controls(self):
        while True:
            try:
                ready, _, _ = select.select([sys.stdin], [], [], 0.2)
            except (OSError, ValueError):
                return

            if not ready:
                continue

            try:
                command = sys.stdin.read(1).lower()
            except OSError:
                return

            if command == "p":
                self.toggle_pause()
            elif command == "q":
                self.request_shutdown()
                return

    def toggle_pause(self):
        with self.lock:
            if self.shutdown_requested:
                return
            now = time.perf_counter()
            if self.run_event.is_set():
                self.run_event.clear()
                self.metrics.paused_at = now
            else:
                if self.metrics.paused_at is not None:
                    self.metrics.total_paused_duration += now - self.metrics.paused_at
                    self.metrics.paused_at = None
                self.run_event.set()

    def request_shutdown(self):
        with self.lock:
            if self.shutdown_requested:
                return

            now = time.perf_counter()
            if self.metrics.paused_at is not None:
                self.metrics.total_paused_duration += now - self.metrics.paused_at
                self.metrics.paused_at = None

            self.shutdown_requested = True
            if self.shutdown_reason is None:
                self.shutdown_reason = "manual"
            self.run_event.set()

    async def wait_for_resume(self):
        while not self.run_event.is_set() and not self.shutdown_requested:
            await asyncio.sleep(0.1)

    async def sleep_with_pause(self, duration_secs: float):
        remaining = duration_secs
        while remaining > 0 and not self.shutdown_requested:
            await self.wait_for_resume()
            if self.shutdown_requested:
                return
            sleep_duration = min(0.1, remaining)
            sleep_started_at = time.perf_counter()
            await asyncio.sleep(sleep_duration)
            remaining -= time.perf_counter() - sleep_started_at

    async def fire_request(self, input_text: str):
        self.metrics.record_request()
        try:
            request_start_time = time.perf_counter()
            stream = await self.client.completions.create(
                model=self.model, prompt=input_text, stream=True
            )
            total_text = ""
            has_not_seen_first_token = True
            ttft: float | None = None
            async for completion in stream:
                chunk_text = "".join(choice.text or "" for choice in completion.choices)
                if chunk_text:
                    total_text += chunk_text
                if has_not_seen_first_token and len(total_text) > 0:
                    ttft = time.perf_counter() - request_start_time
                    self.metrics.add_ttft(ttft)
                    has_not_seen_first_token = False
            request_latency = time.perf_counter() - request_start_time
            self.metrics.record_request_latency(request_latency)
            generated_token_count = len(self.tokenizer.encode(total_text))
            self.metrics.increment_total_tokens(generated_token_count)
            ttft_display = f"{ttft:.3f}s" if ttft is not None else "n/a"
            self.metrics.record_success(
                f"[{time.strftime('%H:%M:%S')}] {generated_token_count} tok in "
                f"{request_latency:.3f}s (TTFT {ttft_display})"
            )
        except Exception as exc:
            self.metrics.record_failure(
                f"[{time.strftime('%H:%M:%S')}] {type(exc).__name__}: {exc}"
            )

    def generate_tokens(self, num_tokens: int) -> str:
        overshoot_chars = num_tokens * 10
        overshooted_text = ""
        token_ids = []
        while len(token_ids) < num_tokens:
            new_text = self.faker.text(max_nb_chars=overshoot_chars)
            new_tokens = self.tokenizer.encode(new_text)
            overshooted_text += new_text
            token_ids.extend(new_tokens)
        tokens = self.tokenizer.encode(overshooted_text)[:num_tokens]
        return self.tokenizer.decode(tokens)

    def snapshot_metrics(self) -> MetricsSnapshot:
        with self.lock:
            now = time.perf_counter()
            paused_duration = self.metrics.total_paused_duration
            if self.metrics.paused_at is not None:
                paused_duration += now - self.metrics.paused_at
            runtime = max(0.0, now - self.metrics.started_at - paused_duration)

            error_rate = (
                self.metrics.total_errors / self.metrics.total_requests * 100
                if self.metrics.total_requests
                else 0.0
            )
            tokens_per_sec = self.metrics.total_tokens / runtime if runtime else 0.0
            avg_ttft = (
                sum(self.metrics.ttfts) / len(self.metrics.ttfts)
                if self.metrics.ttfts
                else 0.0
            )
            avg_request_latency = (
                sum(self.metrics.request_latencies) / len(self.metrics.request_latencies)
                if self.metrics.request_latencies
                else 0.0
            )

            status = "Running"
            if self.shutdown_requested:
                status = "Stopping"
            elif not self.run_event.is_set():
                status = "Paused"

            return MetricsSnapshot(
                captured_at=datetime.now(timezone.utc).isoformat(),
                runtime_seconds=runtime,
                inflight=self.metrics.inflight,
                total_requests=self.metrics.total_requests,
                total_success=self.metrics.total_success,
                total_errors=self.metrics.total_errors,
                total_tokens=self.metrics.total_tokens,
                tokens_per_second=tokens_per_sec,
                avg_ttft_seconds=avg_ttft,
                avg_request_latency_seconds=avg_request_latency,
                error_rate_percent=error_rate,
                status=status,
                recent_successes=list(self.metrics.recent_successes),
                recent_errors=list(self.metrics.recent_errors),
            )

    def start_run_capture(self):
        if self.run_name is None or self.run_capture_thread is not None:
            return
        self.run_capture_thread = Thread(target=self.capture_run_metrics, daemon=False)
        self.run_capture_thread.start()

    def join_run_capture(self):
        if self.run_capture_thread is not None:
            self.run_capture_thread.join()

    def stop_rendering(self):
        self.render_stop_event.set()
        if self.render_thread.is_alive():
            self.render_thread.join(timeout=2)

    def capture_run_metrics(self):
        next_sample_second = 0.0
        while True:
            snapshot = self.snapshot_metrics()

            while (
                snapshot.runtime_seconds >= next_sample_second
                and next_sample_second <= self.RUN_DURATION_SECONDS
            ):
                self.run_samples.append(
                    snapshot.to_timeseries_point(next_sample_second)
                )
                next_sample_second += self.RUN_SAMPLE_INTERVAL_SECONDS

            if snapshot.runtime_seconds >= self.RUN_DURATION_SECONDS:
                with self.lock:
                    if not self.shutdown_requested:
                        self.shutdown_reason = "run_duration_reached"
                self.request_shutdown()
                break

            if self.shutdown_requested:
                break

            time.sleep(0.1)

        self.write_run_metrics_file()

    def write_run_metrics_file(self):
        if self.run_output_path is None or self.run_file_written:
            return

        fieldnames = [
            "run_name",
            "provider",
            "model",
            "traffic_type",
            "traffic_volume",
            "input_size_tokens",
            "requested_duration_seconds",
            "sample_interval_seconds",
            "run_started_at",
            "shutdown_reason",
            "sample_second",
            "captured_at",
            "runtime_seconds",
            "inflight",
            "total_requests",
            "total_success",
            "total_errors",
            "total_tokens",
            "tokens_per_second",
            "avg_ttft_seconds",
            "avg_request_latency_seconds",
            "error_rate_percent",
            "status",
        ]

        self.run_output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.run_output_path.open("w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            for sample in self.run_samples:
                writer.writerow(
                    {
                        "run_name": self.run_name,
                        "provider": self.provider.value,
                        "model": self.model,
                        "traffic_type": self.traffic_type,
                        "traffic_volume": self.traffic_volume,
                        "input_size_tokens": self.input_size_tokens,
                        "requested_duration_seconds": self.RUN_DURATION_SECONDS,
                        "sample_interval_seconds": self.RUN_SAMPLE_INTERVAL_SECONDS,
                        "run_started_at": self.run_started_at,
                        "shutdown_reason": self.shutdown_reason or "manual",
                        **sample,
                    }
                )
        self.run_file_written = True

    async def generate_traffic(
        self, input_num_tokens: int, traffic_volume: int, cadence_seconds: int
    ):
        while not self.shutdown_requested:
            await self.wait_for_resume()
            if self.shutdown_requested:
                break

            requests = []
            for _ in range(traffic_volume):
                if self.shutdown_requested or not self.run_event.is_set():
                    break
                requests.append(
                    asyncio.create_task(
                        self.fire_request(self.generate_tokens(input_num_tokens))
                    )
                )
                await asyncio.sleep(0)

            if requests:
                await asyncio.gather(*requests)

            if self.shutdown_requested:
                break

            await self.sleep_with_pause(cadence_seconds)

    def generate_bursty_traffic(self, input_num_tokens: int, traffic_volume: int):
        self.traffic_type = "bursty"
        self.traffic_volume = traffic_volume
        self.input_size_tokens = input_num_tokens
        self.start_run_capture()
        try:
            asyncio.run(
                self.generate_traffic(
                    input_num_tokens,
                    traffic_volume,
                    TRAFFIC_CADENCE_SECONDS["bursty"],
                )
            )
        finally:
            self.restore_terminal()
            self.join_run_capture()
            self.stop_rendering()

    def generate_constant_traffic(self, input_num_tokens: int, traffic_volume: int):
        self.traffic_type = "constant"
        self.traffic_volume = traffic_volume
        self.input_size_tokens = input_num_tokens
        self.start_run_capture()
        try:
            asyncio.run(
                self.generate_traffic(
                    input_num_tokens,
                    traffic_volume,
                    TRAFFIC_CADENCE_SECONDS["constant"],
                )
            )
        finally:
            self.restore_terminal()
            self.join_run_capture()
            self.stop_rendering()

    def render_box(
        self, title: str, body_lines: List[str], width: int, min_body_lines: int = 0
    ) -> List[str]:
        inner_width = width - 2
        padded_lines = body_lines + [""] * max(0, min_body_lines - len(body_lines))
        return [
            f"┌{'─' * inner_width}┐",
            f"│{title:^{inner_width}}│",
            f"├{'─' * inner_width}┤",
            *[f"│{line[:inner_width]:<{inner_width}}│" for line in padded_lines],
            f"└{'─' * inner_width}┘",
        ]

    def format_log_body_lines(
        self, messages: List[str], width: int, empty_message: str
    ) -> List[str]:
        body_lines: List[str] = []
        for message in messages:
            wrapped_lines = textwrap.wrap(
                message,
                width=width,
                break_long_words=False,
                break_on_hyphens=False,
            ) or [""]
            body_lines.extend(f" {line}" for line in wrapped_lines)
            body_lines.append("")

        if body_lines:
            body_lines.pop()
            return body_lines

        return [f" {empty_message}"]

    def render_dashboard(self):
        while not self.render_stop_event.is_set():
            snapshot = self.snapshot_metrics()
            system("clear")

            rows = [
                ("Status", snapshot.status),
                ("Traffic", format_traffic_profile(self.traffic_type, self.traffic_volume))
                if self.traffic_type is not None and self.traffic_volume is not None
                else ("Traffic", "-"),
                ("Runtime", f"{snapshot.runtime_seconds:.2f}s"),
                ("In-flight", str(snapshot.inflight)),
                ("Total Requests", str(snapshot.total_requests)),
                ("Tokens/s", f"{snapshot.tokens_per_second:.2f}"),
                ("Avg TTFT", f"{snapshot.avg_ttft_seconds:.3f}s"),
                ("Avg Latency", f"{snapshot.avg_request_latency_seconds:.3f}s"),
                ("Successes", str(snapshot.total_success)),
                ("Errors", str(snapshot.total_errors)),
                ("Error Rate", f"{snapshot.error_rate_percent:.2f}%"),
            ]
            if self.run_name is not None and self.run_output_path is not None:
                rows.extend(
                    [
                        ("Run", self.run_name),
                        ("Run File", str(self.run_output_path)),
                    ]
                )

            metric_body_lines = [
                f" {label:<18} {value:>24}" for label, value in rows
            ]
            metric_body_lines.extend(
                [
                    "",
                    " [p] pause/resume  [q] quit gracefully"
                    if self.controls_enabled
                    else " Controls unavailable (non-interactive)",
                ]
            )

            terminal_width = shutil.get_terminal_size((120, 20)).columns
            metrics_width = 46
            gap_width = 2
            min_log_width = 38
            can_render_three_columns = (
                terminal_width
                >= metrics_width + gap_width * 2 + min_log_width * 2
            )
            can_render_two_columns = (
                terminal_width >= metrics_width + gap_width + min_log_width
            )

            if can_render_three_columns:
                remaining_width = terminal_width - metrics_width - gap_width * 2
                success_width = remaining_width // 2
                error_width = remaining_width - success_width
                success_body_lines = self.format_log_body_lines(
                    snapshot.recent_successes,
                    success_width - 4,
                    "No successes recorded.",
                )
                error_body_lines = self.format_log_body_lines(
                    snapshot.recent_errors,
                    error_width - 4,
                    "No errors recorded.",
                )
                min_body_lines = max(
                    len(metric_body_lines),
                    len(success_body_lines),
                    len(error_body_lines),
                )
                metrics_box = self.render_box(
                    "BenchPress Performance Harness",
                    metric_body_lines,
                    metrics_width,
                    min_body_lines=min_body_lines,
                )
                successes_box = self.render_box(
                    "Recent Successes",
                    success_body_lines,
                    success_width,
                    min_body_lines=min_body_lines,
                )
                errors_box = self.render_box(
                    "Recent Errors",
                    error_body_lines,
                    error_width,
                    min_body_lines=min_body_lines,
                )
                print(
                    "\n".join(
                        f"{left}{' ' * gap_width}{middle}{' ' * gap_width}{right}"
                        for left, middle, right in zip(
                            metrics_box, successes_box, errors_box
                        )
                    )
                )
            elif can_render_two_columns:
                activity_width = terminal_width - metrics_width - gap_width
                success_body_lines = self.format_log_body_lines(
                    snapshot.recent_successes,
                    activity_width - 4,
                    "No successes recorded.",
                )
                error_body_lines = self.format_log_body_lines(
                    snapshot.recent_errors,
                    activity_width - 4,
                    "No errors recorded.",
                )
                activity_body_lines = [
                    " Successes",
                    "",
                    *success_body_lines,
                    "",
                    " Errors",
                    "",
                    *error_body_lines,
                ]
                min_body_lines = max(
                    len(metric_body_lines), len(activity_body_lines)
                )
                metrics_box = self.render_box(
                    "BenchPress Performance Harness",
                    metric_body_lines,
                    metrics_width,
                    min_body_lines=min_body_lines,
                )
                activity_box = self.render_box(
                    "Recent Activity",
                    activity_body_lines,
                    activity_width,
                    min_body_lines=min_body_lines,
                )
                print(
                    "\n".join(
                        f"{left}{' ' * gap_width}{right}"
                        for left, right in zip(metrics_box, activity_box)
                    )
                )
            else:
                metrics_box = self.render_box(
                    "BenchPress Performance Harness",
                    metric_body_lines,
                    metrics_width,
                )
                success_body_lines = self.format_log_body_lines(
                    snapshot.recent_successes,
                    metrics_width - 4,
                    "No successes recorded.",
                )
                errors_body_lines = self.format_log_body_lines(
                    snapshot.recent_errors,
                    metrics_width - 4,
                    "No errors recorded.",
                )
                successes_box = self.render_box(
                    "Recent Successes", success_body_lines, metrics_width
                )
                errors_box = self.render_box(
                    "Recent Errors", errors_body_lines, metrics_width
                )
                print(
                    "\n".join(
                        [
                            *metrics_box,
                            "",
                            *successes_box,
                            "",
                            *errors_box,
                        ]
                    )
                )
            if self.render_stop_event.wait(1):
                break


def run_benchmark(
    *,
    provider: Provider,
    model: str,
    traffic_type: str,
    traffic_volume: int,
    input_size_tokens: int,
    run_name: str | None,
    run_output_dir: Path = DEFAULT_RUNS_DIR,
) -> RequestGenerator:
    request_generator = RequestGenerator(
        model=model,
        provider=provider,
        run_name=run_name,
        run_output_dir=run_output_dir,
    )

    if traffic_type == "constant":
        request_generator.generate_constant_traffic(input_size_tokens, traffic_volume)
    elif traffic_type == "bursty":
        request_generator.generate_bursty_traffic(input_size_tokens, traffic_volume)
    else:
        raise ValueError(f"Unsupported traffic type: {traffic_type}")

    return request_generator


def run_experiment(config: ExperimentConfig) -> ExperimentResult:
    run_output_paths: List[Path] = []
    total_runs = len(config.runs)
    for index, run in enumerate(config.runs, start=1):
        print(
            f"Starting experiment run {index}/{total_runs}: {run.name} "
            f"({format_traffic_profile(run.traffic_type, run.traffic_volume)}, "
            f"{run.input_size_tokens} toks)"
        )
        request_generator = run_benchmark(
            provider=run.provider,
            model=run.model,
            traffic_type=run.traffic_type,
            traffic_volume=run.traffic_volume,
            input_size_tokens=run.input_size_tokens,
            run_name=run.name,
            run_output_dir=config.runs_dir,
        )

        if request_generator.run_output_path is None or not request_generator.run_file_written:
            raise RuntimeError(f"Experiment run {run.name!r} did not produce a run CSV")

        run_output_paths.append(request_generator.run_output_path)
        print(f"Completed run {run.name}; wrote {request_generator.run_output_path}")

        if index < total_runs and run.prompt_after_run is not None:
            wait_for_keypress(run.prompt_after_run)

    comparison_output_path: Path | None = None
    if config.comparison.enabled:
        comparison_output_path = compare_run_files(
            run_files=run_output_paths,
            metrics=config.comparison.metrics,
            output_path=config.comparison.output_path,
            title=config.comparison.title,
        )

    return ExperimentResult(
        run_output_paths=run_output_paths,
        comparison_output_path=comparison_output_path,
    )


def experiment_cli(argv: Sequence[str] | None = None) -> int:
    args = build_experiment_parser().parse_args(
        list(argv) if argv is not None else None
    )
    result = run_experiment(load_experiment_config(args.config))
    if result.comparison_output_path is not None:
        print(f"Wrote experiment comparison plot to {result.comparison_output_path}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "experiment":
        result = run_experiment(load_experiment_config(args.config))
        if result.comparison_output_path is not None:
            print(f"Wrote experiment comparison plot to {result.comparison_output_path}")
        return 0
    if args.command == "compare-runs":
        output_path = compare_runs(
            runs_dir=args.runs_dir,
            glob_pattern=args.glob,
            metrics=args.metrics,
            output_path=args.output,
            title=args.title,
        )
        print(f"Wrote run comparison plot to {output_path}")
        return 0

    traffic_volume = resolve_traffic_volume(
        args.traffic_volume,
        traffic_type=args.traffic_type,
        field_name="traffic_volume",
    )
    request_generator = run_benchmark(
        provider=Provider(args.provider),
        model=args.model,
        traffic_type=args.traffic_type,
        traffic_volume=traffic_volume,
        input_size_tokens=args.input_size_tokens,
        run_name=args.run,
    )

    if request_generator.run_output_path is not None and request_generator.run_file_written:
        print(f"Run metrics written to {request_generator.run_output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
