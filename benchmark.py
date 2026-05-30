import argparse
import asyncio
import time
from dataclasses import dataclass
from os import environ, system
from threading import Lock, Thread

from faker import Faker
from openai import AsyncOpenAI
from transformers import AutoTokenizer


def arg_parser():
    parser = argparse.ArgumentParser(description="VLLM benchmark harness")
    parser.add_argument(
        "--traffic-type", type=str, choices=["constant", "bursty"], default="constant"
    )
    parser.add_argument("--input-size-tokens", type=int, default=100)
    return parser.parse_args()


@dataclass
class Metrics:
    started_at: float
    inflight: int
    total_requests: int
    total_success: int
    total_errors: int
    lock: Lock
    avg_tokens_per_sec: int

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


class RequestGenerator:
    def __init__(self):
        api_key, base_url = environ.get("VLLM_API_KEY"), environ.get("VLLM_BASE_URL")
        if not api_key or not base_url:
            raise Exception("VLLM_API_KEY and VLLM_BASE_URL must be set")
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.faker = Faker()
        self.tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-3.1-8B-Instruct"
        )
        self.lock = Lock()
        self.metrics = Metrics(
            started_at=time.time(),
            inflight=0,
            total_requests=0,
            total_success=0,
            total_errors=0,
            lock=self.lock,
            avg_tokens_per_sec=0,
        )
        Thread(target=self.render_dashboard, daemon=True).start()

    async def fire_request(self, input_text: str):
        self.metrics.record_request()
        try:
            response = await self.client.completions.create(
                model="meta-llama/Llama-3.1-8B-Instruct", prompt=input_text
            )
            self.metrics.record_success()
            return response
        except:
            self.metrics.record_failure()

    def generate_tokens(self, num_tokens: int) -> str:
        overshoot_chars = num_tokens * 10
        overshooted_text = ""
        token_ids = []
        while len(token_ids) < num_tokens:
            new_text = self.faker.text(max_nb_chars=overshoot_chars)
            new_tokens = self.tokenizer.encode(new_text, add_special_tokens=False)
            overshooted_text += new_text
            token_ids.extend(new_tokens)
        tokens = self.tokenizer.encode(overshooted_text, add_special_tokens=False)[
            :num_tokens
        ]
        return self.tokenizer.decode(tokens)

    async def generate_traffic(
        self, input_num_tokens: int, max_burst: int, cool_down_secs: int
    ):
        while True:
            requests = [
                self.fire_request(self.faker.text(max_nb_chars=input_num_tokens))
                for _ in range(max_burst)
            ]
            await asyncio.gather(*requests)
            await asyncio.sleep(cool_down_secs)

    def generate_bursty_traffic(self, inputNumChars: int):
        asyncio.run(self.generate_traffic(inputNumChars, 10, 10))

    def generate_constant_traffic(self, inputNumChars: int):
        asyncio.run(self.generate_traffic(inputNumChars, 1, 1))

    def render_dashboard(self):
        while True:
            with self.lock:
                runtime = time.time() - self.metrics.started_at

                error_rate = (
                    self.metrics.total_errors / self.metrics.total_requests * 100
                    if self.metrics.total_requests
                    else 0.0
                )

                system("clear")

                print(
                    f"""
            ┌──────────────────────────────────────────────┐
            │ vLLM Benchmark Harness                       │
            ├──────────────────────────────────────────────┤
            │ Runtime            {runtime:<25}│
            ├──────────────────────────────────────────────┤
            │ In-flight          {self.metrics.inflight:<25}│
            │ Total Requests     {self.metrics.total_requests:<25}│
            │ Successes          {self.metrics.total_success:<25}│
            │ Errors             {self.metrics.total_errors:<25}│
            │ Error Rate         {f"{error_rate:.2f}%":<25}│
            └──────────────────────────────────────────────┘
            """.strip()
                )
                time.sleep(1)


if __name__ == "__main__":
    args = arg_parser()
    request_generator = RequestGenerator()

    if args.traffic_type == "constant":
        request_generator.generate_constant_traffic(args.input_size_chars)
    elif args.traffic_type == "bursty":
        request_generator.generate_bursty_traffic(args.input_size_chars)
