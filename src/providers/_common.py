"""Helpers shared by the provider clients."""

from __future__ import annotations

import asyncio
from collections.abc import Callable

from src.utils.logging import logger


async def record_api_call(write: Callable[[], None]) -> None:
    """Run a blocking `api_calls` write on a worker thread.

    `write` is a zero-argument thunk — typically ``functools.partial`` over
    `log_api_call`. Failing to record cost must never mask the API result, or
    the API error, it describes, so any exception from the write is swallowed
    with a warning rather than propagated.
    """
    try:
        await asyncio.to_thread(write)
    except Exception as exc:
        logger.warning("could not record api_call: {}: {}", type(exc).__name__, exc)
