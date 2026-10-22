"""Centralised application configuration for the MATLAB Troubleshooting AI.

Loads environment variables from a ``.env`` file once at import time and
exposes all runtime settings through a frozen, cached ``Settings`` dataclass.

All other modules must import from this module instead of reading
``os.environ`` directly.

Example:
    >>> from config.settings import get_settings
    >>> cfg = get_settings()
    >>> print(cfg.gemini_model)
    gemini-2.5-flash
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

_PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env", override=False)

logger: logging.Logger = logging.getLogger(__name__)


def _require_env(key: str) -> str:
    """Read a required environment variable.

    Args:
        key: Name of the environment variable.

    Returns:
        The string value of the environment variable.

    Raises:
        ValueError: If the variable is missing or empty.
    """
    value: str | None = os.getenv(key)
    if not value:
        raise ValueError(
            f"Required environment variable '{key}' is not set. "
            "Copy .env.example to .env and add your key."
        )
    return value


def _configure_logging(level: str) -> None:
    """Configure the root logger with a structured format.

    Safe to call multiple times; adds a handler only when none exist.

    Args:
        level: Standard Python log level string (e.g. ``INFO``).
    """
    numeric_level: int = getattr(logging, level.upper(), logging.INFO)
    root: logging.Logger = logging.getLogger()

    if not root.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        root.addHandler(handler)

    root.setLevel(numeric_level)


@dataclass(frozen=True)
class Settings:
    """Immutable application configuration loaded from the environment."""

    gemini_api_key: str
    tavily_api_key: str
    log_level: str

    gemini_model: str
    embedding_model: str

    project_root: Path = field(default_factory=lambda: _PROJECT_ROOT)

    knowledge_base_dir: Path = field(
        default_factory=lambda: _PROJECT_ROOT / "knowledge_base" / "pdfs"
    )

    faiss_index_dir: Path = field(
        default_factory=lambda: _PROJECT_ROOT / "faiss_index"
    )

    uploads_dir: Path = field(
        default_factory=lambda: _PROJECT_ROOT / "uploads" / "images"
    )

    def ensure_runtime_dirs(self) -> None:
        """Create runtime directories if they do not already exist."""
        for directory in (
            self.faiss_index_dir,
            self.uploads_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the application Settings singleton."""

    log_level: str = os.getenv("LOG_LEVEL", "INFO").upper()
    _configure_logging(log_level)

    settings = Settings(
        gemini_api_key=_require_env("GEMINI_API_KEY"),
        tavily_api_key=_require_env("TAVILY_API_KEY"),
        gemini_model=_require_env("GEMINI_MODEL"),
        embedding_model=_require_env("EMBEDDING_MODEL"),
        log_level=log_level,
    )

    settings.ensure_runtime_dirs()

    return settings