import argparse
import asyncio
import atexit
import shutil
import select
import sys
import termios
import textwrap
import time
from dataclasses import dataclass, field
from os import environ, system
from threading import Event, Lock, Thread
from typing import List
from urllib.parse import urlsplit, urlunsplit

from faker import Faker
from openai import AsyncOpenAI

from contracts import Provider
from tokenizer import Tokenizer


def arg_parser():
    parser = argparse.ArgumentParser(
        description="BenchPress a light weight LLM endpoint benchmark harness"
    )
    parser.add_argument(
        "--traffic-type", type=str, choices=["constant", "bursty"], default="constant"
    )
    parser.add_argument("--input-size-tokens", type=int, default=100)
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument(
        "--provider", type=str, choices=["vllm", "openai"], default="vllm"
    )
    return parser.parse_args()


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


class RequestGenerator:
    def __init__(self, model: str, provider: Provider):
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
        self.tokenizer = Tokenizer(model, provider)
        self.lock = Lock()
        self.run_event = Event()
        self.run_event.set()
        self.shutdown_requested = False
        self.controls_enabled = False
        self.stdin_fd = None
        self.original_terminal_settings = None
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
        Thread(target=self.render_dashboard, daemon=True).start()

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

    async def generate_traffic(
        self, input_num_tokens: int, max_burst: int, cool_down_secs: int
    ):
        while not self.shutdown_requested:
            await self.wait_for_resume()
            if self.shutdown_requested:
                break

            requests = []
            for _ in range(max_burst):
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

            await self.sleep_with_pause(cool_down_secs)

    def generate_bursty_traffic(self, input_num_tokens: int):
        try:
            asyncio.run(self.generate_traffic(input_num_tokens, 10, 10))
        finally:
            self.restore_terminal()

    def generate_constant_traffic(self, input_num_tokens: int):
        try:
            asyncio.run(self.generate_traffic(input_num_tokens, 1, 1))
        finally:
            self.restore_terminal()

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
        while True:
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

                system("clear")

                status = "Running"
                if self.shutdown_requested:
                    status = "Stopping"
                elif not self.run_event.is_set():
                    status = "Paused"

                rows = [
                    ("Status", status),
                    ("Runtime", f"{runtime:.2f}s"),
                    ("In-flight", str(self.metrics.inflight)),
                    ("Total Requests", str(self.metrics.total_requests)),
                    ("Tokens/s", f"{tokens_per_sec:.2f}"),
                    ("Avg TTFT", f"{avg_ttft:.3f}s"),
                    ("Successes", str(self.metrics.total_success)),
                    ("Errors", str(self.metrics.total_errors)),
                    ("Error Rate", f"{error_rate:.2f}%"),
                ]
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
                        self.metrics.recent_successes,
                        success_width - 4,
                        "No successes recorded.",
                    )
                    error_body_lines = self.format_log_body_lines(
                        self.metrics.recent_errors,
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
                        self.metrics.recent_successes,
                        activity_width - 4,
                        "No successes recorded.",
                    )
                    error_body_lines = self.format_log_body_lines(
                        self.metrics.recent_errors,
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
                        self.metrics.recent_successes,
                        metrics_width - 4,
                        "No successes recorded.",
                    )
                    errors_body_lines = self.format_log_body_lines(
                        self.metrics.recent_errors,
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
            time.sleep(1)


if __name__ == "__main__":
    args = arg_parser()
    provider = Provider(args.provider)
    request_generator = RequestGenerator(model=args.model, provider=provider)

    if args.traffic_type == "constant":
        request_generator.generate_constant_traffic(args.input_size_tokens)
    elif args.traffic_type == "bursty":
        request_generator.generate_bursty_traffic(args.input_size_tokens)
