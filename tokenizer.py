from typing import List

import tiktoken
from transformers import AutoTokenizer

from contracts import Provider


class Tokenizer:
    def __init__(self, model: str, provider: Provider):
        self.provider = provider
        if provider == Provider.OPENAI:
            self.tiktoken_tokenizer = tiktoken.encoding_for_model(model)
        else:
            self.transformers_tokenizer = AutoTokenizer.from_pretrained(model)

    def encode(self, text: str) -> List[int]:
        if self.provider == Provider.OPENAI:
            return self.tiktoken_tokenizer.encode(text)
        else:
            return self.transformers_tokenizer.encode(text, add_special_tokens=False)

    def decode(self, tokens: List[int]) -> str:
        if self.provider == Provider.OPENAI:
            return self.tiktoken_tokenizer.decode(tokens)
        else:
            return self.transformers_tokenizer.decode(tokens, add_special_tokens=False)
