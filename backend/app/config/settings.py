"""
Configuration management using Pydantic Settings.

All configurable values are centralised here and read from
environment variables (or a .env file).  No magic numbers
should exist elsewhere in the codebase.
"""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


# Resolve project root so default paths work regardless of cwd.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # backend/


class Settings(BaseSettings):
    """Application settings — populated from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ── API ──────────────────────────────────────────────────
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    DEBUG: bool = True
    CORS_ORIGINS: list[str] = ["http://localhost:3000"]

    # ── Database ─────────────────────────────────────────────
    DATABASE_URL: str = f"sqlite+aiosqlite:///{_PROJECT_ROOT / 'data' / 'search_engine.db'}"

    # ── Crawler (placeholder — used in later phases) ─────────
    CRAWLER_USER_AGENT: str = "SearchEngineBot/0.1 (+https://github.com/search-engine)"
    CRAWLER_TIMEOUT: int = 10          # seconds
    CRAWLER_DELAY: float = 1.0         # seconds between requests
    MAX_CRAWL_PAGES: int = 100
    MAX_CRAWL_DEPTH: int = 3
    CRAWLER_MAX_RESPONSE_BYTES: int = 5 * 1024 * 1024  # 5 MB

    # ── Ranking (placeholder — used in later phases) ─────────
    BM25_K1: float = 1.2
    BM25_B: float = 0.75
    BM25_TITLE_WEIGHT: float = 2.0
    BM25_BODY_WEIGHT: float = 1.0

    # ── Index ────────────────────────────────────────────────
    INDEX_PATH: str = str(_PROJECT_ROOT / "data" / "index")

    # ── Snippets ─────────────────────────────────────────────
    SNIPPET_LENGTH: int = 160
    SNIPPET_MIN_LENGTH: int = 80

    # ── Phrase search ────────────────────────────────────────
    PHRASE_BOOST: float = 2.0

    # ── Query processing ─────────────────────────────────────
    QUERY_MAX_WEIGHT: float = 3.0


# Singleton — import this everywhere instead of constructing Settings().
settings = Settings()
