"""Shared retry policy — the tenacity backoff every provider client uses.

Backoff is exponential (1s, 2s, 4s, 8s ... capped at 16s) over at most
`MAX_ATTEMPTS` tries. Each provider passes its own `is_retryable` predicate:
transient faults (timeouts, connection errors, HTTP 429 and 5xx) are retried;
everything else (4xx auth, content policy, bad request) propagates on the
first raise. Every retry is logged with its cause via `before_sleep`.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from tenacity import (
    AsyncRetrying,
    RetryCallState,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from src.utils.logging import logger

# The spec calls for "1s, 2s, 4s, 8s, 16s — max 5 attempts". Five attempts
# means four waits (1, 2, 4, 8s) — the start of that exponential schedule.
MAX_ATTEMPTS = 5

_SleepFn = Callable[[float], Awaitable[None]]


async def _default_async_sleep(seconds: float) -> None:
    """The pause tenacity takes between attempts (injectable for tests)."""
    await asyncio.sleep(seconds)


def _make_before_sleep(label: str) -> Callable[[RetryCallState], None]:
    """Build a `before_sleep` callback that logs each retry with its cause."""

    def _log(retry_state: RetryCallState) -> None:
        outcome = retry_state.outcome
        exc = outcome.exception() if outcome is not None else None
        action = retry_state.next_action
        delay = action.sleep if action is not None else 0.0
        logger.warning(
            "{} call failed on attempt {} ({}: {}); retrying in {:.0f}s",
            label,
            retry_state.attempt_number,
            type(exc).__name__ if exc is not None else "unknown",
            exc,
            delay,
        )

    return _log


def make_async_retrying(
    is_retryable: Callable[[BaseException], bool],
    *,
    label: str,
    max_attempts: int = MAX_ATTEMPTS,
    sleep: _SleepFn = _default_async_sleep,
) -> AsyncRetrying:
    """An `AsyncRetrying` driver: retry transient errors, log every retry.

    Use it as ``async for attempt in make_async_retrying(...): with attempt:``.
    `reraise=True` means the final failure surfaces as the original exception
    rather than tenacity's `RetryError` wrapper.
    """
    return AsyncRetrying(
        sleep=sleep,
        retry=retry_if_exception(is_retryable),
        wait=wait_exponential(multiplier=1, min=1, max=16),
        stop=stop_after_attempt(max_attempts),
        before_sleep=_make_before_sleep(label),
        reraise=True,
    )
