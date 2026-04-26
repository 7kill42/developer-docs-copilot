from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CHROMA_DIR = DATA_DIR / "chroma"

load_dotenv(BASE_DIR / ".env")


@dataclass(frozen=True)
class Settings:
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_base_url: str = os.getenv(
        "OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    openai_chat_model: str = os.getenv("OPENAI_CHAT_MODEL", "qwen3.6-flash")
    openai_embedding_model: str = os.getenv(
        "OPENAI_EMBEDDING_MODEL", "text-embedding-v4"
    )
    chroma_path: Path = Path(os.getenv("CHROMA_PERSIST_DIR", str(CHROMA_DIR)))
    chroma_collection_name: str = os.getenv(
        "CHROMA_COLLECTION_NAME", "sqlalchemy_docs"
    )
    docs_base_url: str = os.getenv("DOCS_BASE_URL", "https://docs.sqlalchemy.org/en/20/")
    max_seed_pages: int = int(os.getenv("MAX_SEED_PAGES", "14"))
    chunk_token_target: int = int(os.getenv("CHUNK_TOKEN_TARGET", "450"))
    top_k: int = int(os.getenv("TOP_K", "6"))
    retrieve_k: int = int(os.getenv("RETRIEVE_K", "18"))
    min_relevance_score: float = float(os.getenv("MIN_RELEVANCE_SCORE", "0.22"))
    model_context_k: int = int(os.getenv("MODEL_CONTEXT_K", "3"))
    model_context_char_limit: int = int(os.getenv("MODEL_CONTEXT_CHAR_LIMIT", "900"))
    openai_timeout_seconds: float = float(os.getenv("OPENAI_TIMEOUT_SECONDS", "30"))

    @property
    def has_openai_key(self) -> bool:
        return bool(self.openai_api_key.strip())


settings = Settings()
