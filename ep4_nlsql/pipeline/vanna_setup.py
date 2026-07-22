"""Vanna NL-to-SQL setup: CatalogVanna class, singleton, and training bootstrap.

This version uses the generic Vanna base classes plus a local OpenAI-compatible
HTTP endpoint, which works with Ollama's OpenAI-compatible API surface.

Run once after creating the database:
    python -m ep4_nlsql.pipeline.vanna_setup
"""
from __future__ import annotations

import functools
import os
from pathlib import Path

from dotenv import load_dotenv
from vanna.legacy.chromadb.chromadb_vector import ChromaDB_VectorStore
from vanna.openai.openai_chat import OpenAI_Chat

from ep4_nlsql.data.schema import DDL, QA_PAIRS

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# Ollama exposes an OpenAI-compatible endpoint by default.
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "ollama")  # any non-empty value is fine
_CHROMA_PATH = Path(__file__).resolve().parent.parent / "data" / "chroma"


class CatalogVanna(ChromaDB_VectorStore, OpenAI_Chat):
    """Vanna instance backed by ChromaDB + a local OpenAI-compatible LLM."""

    def __init__(self, config: dict) -> None:
        ChromaDB_VectorStore.__init__(self, config=config)
        OpenAI_Chat.__init__(self, config=config)


@functools.lru_cache(maxsize=1)
def get_vanna() -> CatalogVanna:
    """Return the shared CatalogVanna singleton (created once, then cached)."""
    _CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    config = {
        "api_key": OLLAMA_API_KEY,
        "base_url": OLLAMA_BASE_URL,
        "model": OLLAMA_MODEL,
        "path": str(_CHROMA_PATH),
    }
    return CatalogVanna(config=config)


def train() -> None:
    """Train Vanna on schema DDL and Q&A pairs. Run once."""
    vn = get_vanna()

    vn.train(ddl=DDL)
    for pair in QA_PAIRS:
        vn.train(question=pair["question"], sql=pair["sql"])

    print(f"Trained Vanna on {len(QA_PAIRS)} Q&A pairs")
    print(f"ChromaDB stored at: {_CHROMA_PATH}")
    print(f"LLM endpoint: {OLLAMA_BASE_URL}")
    print(f"LLM model: {OLLAMA_MODEL}")


if __name__ == "__main__":
    train()
