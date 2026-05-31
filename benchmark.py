import argparse
import asyncio
import time
from dataclasses import dataclass, field
from itertools import chain
from os import environ, system
from threading import Lock, Thread
from typing import List

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
    ttfts: List[float] = field(default_factory=list)

    def record_request(self):
        with self.lock:
            self.inflight += 1
            self.total_requests += 1

    def record_success(self):
        with self.lock:
            self.inflight -= 1
            self.total_success += 1

    def record_failure(self):
        with self.lock:
            self.inflight -= 1
            self.total_errors += 1

    def increment_total_tokens(self, tokens: int):
        with self.lock:
            self.total_tokens += tokens

    def add_ttft(self, ttft: float):
        with self.lock:
            self.ttfts.append(ttft)


class RequestGenerator:
    def __init__(self, model: str, provider: Provider):
        api_key, base_url = environ.get("VLLM_API_KEY"), environ.get("VLLM_BASE_URL")
        if not api_key:
            raise Exception("You must set VLLM_API_KEY")
        if provider == Provider.VLLM and not base_url:
            raise Exception("You must set VLLM_BASE_URL")
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.faker = Faker()
        self.model = model
        self.tokenizer = Tokenizer(model, provider)
        self.lock = Lock()
        self.metrics = Metrics(
            started_at=time.perf_counter(),
            inflight=0,
            total_requests=0,
            total_success=0,
            total_errors=0,
            lock=self.lock,
            total_tokens=0,
        )
        Thread(target=self.render_dashboard, daemon=True).start()

    async def fire_request(self, input_text: str):
        self.metrics.record_request()
        try:
            request_start_time = time.perf_counter()
            stream = await self.client.completions.create(
                model=self.model, prompt=input_text, stream=True
            )
            total_text = ""
            has_not_seen_first_token = True
            async for completion in stream:
                chunk_text = "".join(
                    chain.from_iterable(map(lambda c: c.text, completion.choices))
                )
                if chunk_text:
                    total_text += chunk_text
                if has_not_seen_first_token and len(total_text) > 0:
                    self.metrics.add_ttft(time.perf_counter() - request_start_time)
                    has_not_seen_first_token = False
            generated_tokens = self.tokenizer.encode(total_text)
            self.metrics.increment_total_tokens(len(generated_tokens))
            self.metrics.record_success()
        except:
            self.metrics.record_failure()

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
        while True:
            requests = [
                self.fire_request(self.generate_tokens(input_num_tokens))
                for _ in range(max_burst)
            ]
            await asyncio.gather(*requests)
            await asyncio.sleep(cool_down_secs)

    def generate_bursty_traffic(self, input_num_tokens: int):
        asyncio.run(self.generate_traffic(input_num_tokens, 10, 10))

    def generate_constant_traffic(self, input_num_tokens: int):
        asyncio.run(self.generate_traffic(input_num_tokens, 1, 1))

    def render_dashboard(self):
        while True:
            with self.lock:
                runtime = time.perf_counter() - self.metrics.started_at

                error_rate = (
                    self.metrics.total_errors / self.metrics.total_requests * 100
                    if self.metrics.total_requests
                    else 0.0
                )

                tokens_per_sec = self.metrics.total_tokens / runtime

                avg_ttft = (
                    sum(self.metrics.ttfts) / len(self.metrics.ttfts)
                    if self.metrics.ttfts
                    else 0.0
                )

                system("clear")

                print(
                    f"""
            ┌──────────────────────────────────────────────┐
            │ BenchPress Performance Harness               │
            ├──────────────────────────────────────────────┤
            │ Runtime            {runtime:<25}│
            ├──────────────────────────────────────────────┤
            │ In-flight          {self.metrics.inflight:<25}│
            │ Total Requests     {self.metrics.total_requests:<25}│
            │ Tokens/s       {tokens_per_sec:<25}│
            │ Avg TTFT (client observed)     {avg_ttft:<25}│
            │ Successes          {self.metrics.total_success:<25}│
            │ Errors             {self.metrics.total_errors:<25}│
            │ Error Rate         {f"{error_rate:.2f}%":<25}│
            └──────────────────────────────────────────────┘
            """.strip()
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
