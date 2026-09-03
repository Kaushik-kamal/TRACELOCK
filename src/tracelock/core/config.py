"""Typed configuration loaded from environment / .env.

Config errors must be distinguishable from viability failures.  A missing API
key is a setup problem, NOT evidence that dynamic discovery is impossible, and
the gate script relies on that distinction to avoid a false negative.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="TL_",
        extra="ignore",
    )

    serpapi_api_key: str = ""
    serp_engine: str = "google_lens"
    max_candidates: int = Field(default=20, ge=1, le=100)
    http_timeout: float = Field(default=30.0, gt=0)
    data_dir: Path = Path("data")
    upload_retention: str = "1h"

    @property
    def runs_dir(self) -> Path:
        return self.data_dir / "runs"


def load_settings() -> Settings:
    """Load settings. Kept as a function so tests can construct their own."""
    return Settings()
