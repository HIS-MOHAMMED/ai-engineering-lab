"""Vanna NL-to-SQL setup: CatalogVanna class, singleton, and training bootstrap.

This version avoids Vanna backend-specific imports and talks to a local
Ollama server through its OpenAI-compatible HTTP API.

Run once after creating the database:
    python -m ep4_nlsql.pipeline.vanna_setup
"""
from __future__ import annotations

import functools
import os
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
from vanna.legacy.chromadb.chromadb_vector import ChromaDB_VectorStore

from ep4_nlsql.data.schema import DDL, QA_PAIRS

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "ollama")
_CHROMA_PATH = Path(__file__).resolve().parent.parent / "data" / "chroma"


class OllamaOpenAICompatibleChat:
    """Minimal Vanna chat adapter for OpenAI-compatible local models.

    Vanna versions differ quite a bit across releases, so this class intentionally
    avoids importing any model-specific Vanna chat backend. It only provides the
    small surface used by Vanna's generation code: an LLM callable that returns
    a chat completion response.
    """

    def __init__(self, config: dict) -> None:
        self.config = config
        self.api_key = config.get("api_key", OLLAMA_API_KEY)
        self.base_url = config.get("base_url", OLLAMA_BASE_URL).rstrip("/")
        self.model = config.get("model", OLLAMA_MODEL)

    def _chat_messages(self, prompt: str) -> list[dict[str, str]]:
        return [
            {
                "role": "user",
                "content": prompt,
            }
        ]

    def _post_chat(self, prompt: str, **kwargs: Any) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": self._chat_messages(prompt),
            "temperature": kwargs.get("temperature", 0),
        }

        # Ask for a plain text completion from the local OpenAI-compatible API.
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        with httpx.Client(timeout=120) as client:
            response = client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()

        return data["choices"][0]["message"]["content"]

    def submit_prompt(self, prompt: str, **kwargs: Any) -> str:
        return self._post_chat(prompt, **kwargs)

    # Common aliases used by different Vanna versions / integrations.
    def system_message(self, message: str) -> str:
        return message

    def llm(self, prompt: str, **kwargs: Any) -> str:
        return self._post_chat(prompt, **kwargs)

    def generate_sql(self, question: str, **kwargs: Any) -> str:
        return self._post_chat(question, **kwargs)


class CatalogVanna(ChromaDB_VectorStore, OllamaOpenAICompatibleChat):
    """Vanna instance backed by ChromaDB + a local Ollama endpoint."""

    def __init__(self, config: dict) -> None:
        ChromaDB_VectorStore.__init__(self, config=config)
        OllamaOpenAICompatibleChat.__init__(self, config=config)


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
