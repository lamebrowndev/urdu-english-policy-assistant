"""Token counting with the bge-m3 tokenizer."""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache

from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

TokenCounter = Callable[[str], int]


@lru_cache(maxsize=4)
def _load_tokenizer(repo: str, revision: str) -> Tokenizer:
    path = hf_hub_download(repo_id=repo, filename="tokenizer.json", revision=revision)
    return Tokenizer.from_file(path)


def load_bge_m3_counter(repo: str, revision: str) -> TokenCounter:
    """Return a counter backed by the pinned bge-m3 XLM-RoBERTa tokenizer.

    Chunks are capped at 350 tokens so that a Phase 2 query plus title/heading
    plus chunk stays inside the 512-token default window of the
    bge-reranker-v2-m3 reranker, which uses the same XLM-RoBERTa tokenizer.
    """
    tokenizer = _load_tokenizer(repo, revision)

    def count_tokens(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False).ids)

    return count_tokens
