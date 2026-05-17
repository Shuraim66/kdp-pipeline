"""Application settings, loaded from environment variables and `.env`."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed configuration for the pipeline.

    `database_url` is the only required value. Provider API keys are optional
    here and validated at point of use by the code that needs them (Phase 4+),
    so database-only commands run without Fal/Anthropic credentials present.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    database_url: str

    anthropic_api_key: str | None = None
    fal_key: str | None = None

    output_dir: Path = Path("output")
    log_level: str = "INFO"
    log_format: str = "pretty"

    fal_max_concurrent: int = 5
    anthropic_max_concurrent: int = 3

    # Optional ceiling — `build` aborts if a book's projected cost exceeds it.
    max_book_cost_usd: float | None = None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide `Settings` singleton."""
    # pydantic-settings reads `database_url` (and the rest) from the
    # environment, so no constructor arguments are passed here.
    return Settings()  # type: ignore[call-arg]
