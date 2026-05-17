"""Loguru configuration — pretty output in development, JSON in production.

`configure_logging()` installs a single stderr sink, honouring `LOG_LEVEL`
and `LOG_FORMAT` from settings. `diagnose` is left off so loguru never
expands variable values into tracebacks — that could otherwise leak an API
key. If settings cannot be loaded (e.g. no `.env`), sane defaults are used so
logging never blocks a command.
"""

from __future__ import annotations

import sys

from loguru import logger

from src.settings import get_settings

_configured = False


def configure_logging() -> None:
    """Install the stderr log sink from `LOG_LEVEL` / `LOG_FORMAT`. Idempotent."""
    global _configured
    if _configured:
        return
    level = "INFO"
    as_json = False
    try:
        settings = get_settings()
        level = settings.log_level.strip().upper()
        as_json = settings.log_format.strip().lower() == "json"
    except Exception as exc:  # missing .env etc. — fall back to defaults
        logger.debug("using default logging config ({}): {}", type(exc).__name__, exc)
    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        serialize=as_json,
        backtrace=False,
        diagnose=False,
    )
    _configured = True


__all__ = ["configure_logging", "logger"]
