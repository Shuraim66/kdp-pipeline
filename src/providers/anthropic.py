"""Anthropic Messages API client — metadata generation, retried, cost-tracked.

`AnthropicProvider` wraps the async SDK. The SDK's own retry loop is disabled
(`max_retries=0`) so the shared tenacity policy owns retries — that keeps
every attempt visible in the logs and in `api_calls`. `generate_json` layers a
JSON-only instruction on top of `generate_text` for structured calls.
"""

from __future__ import annotations

import asyncio
import base64
import json
import time
from dataclasses import dataclass
from decimal import Decimal
from functools import partial
from typing import Any
from uuid import UUID

import anthropic

from src.db.repos.api_calls import log_api_call
from src.providers._common import record_api_call
from src.settings import get_settings
from src.utils.logging import logger
from src.utils.retries import make_async_retrying

# The spec names `claude-sonnet-4-7`, which is not a real model id. The current
# Sonnet generation is 4.6; metadata generation is a light task for which
# Sonnet is the cost-appropriate tier. Pass `model=` to a call to override.
DEFAULT_MODEL = "claude-sonnet-4-6"

# Anthropic list price per million tokens — (input, output) — as of 2026-05.
_PRICING_PER_MTOK: dict[str, tuple[Decimal, Decimal]] = {
    "claude-sonnet-4-6": (Decimal("3.00"), Decimal("15.00")),
    "claude-opus-4-7": (Decimal("5.00"), Decimal("25.00")),
    "claude-haiku-4-5": (Decimal("1.00"), Decimal("5.00")),
}

_PROVIDER = "anthropic"
_ATTEMPT_TIMEOUT_S = 60.0
_JSON_INSTRUCTION = (
    "Respond with one valid JSON object and nothing else: no prose, no "
    "explanation, and no Markdown code fences."
)


class AnthropicProviderError(RuntimeError):
    """The model returned an unusable response (empty, or not valid JSON)."""


@dataclass(frozen=True, slots=True)
class AnthropicResult:
    """The outcome of one successful Messages API call."""

    text: str
    input_tokens: int
    output_tokens: int
    cost_usd: Decimal
    duration_s: float
    stop_reason: str | None


def cost_for_tokens(model: str, input_tokens: int, output_tokens: int) -> Decimal:
    """Anthropic's charge for a call of the given model and token counts."""
    rate = _PRICING_PER_MTOK.get(model)
    if rate is None:
        logger.warning("no price for model {!r}; recording cost as $0", model)
        return Decimal(0)
    input_rate, output_rate = rate
    cost = (input_rate * input_tokens + output_rate * output_tokens) / 1_000_000
    return cost.quantize(Decimal("0.00001"))


def _is_retryable(exc: BaseException) -> bool:
    """Transient Anthropic faults worth retrying; 4xx (bar 429) are not."""
    if isinstance(exc, anthropic.APIStatusError):
        # 429 rate limit, 5xx server errors, 529 overloaded.
        return exc.status_code == 429 or exc.status_code >= 500
    # APIConnectionError covers APITimeoutError; TimeoutError covers wait_for.
    return isinstance(exc, anthropic.APIConnectionError | TimeoutError)


def _text_of(message: anthropic.types.Message) -> str:
    """Concatenate every text block of a response message."""
    return "".join(
        block.text for block in message.content if isinstance(block, anthropic.types.TextBlock)
    )


def _strip_code_fence(text: str) -> str:
    """Drop a leading ```/```json fence and its closing ``` if present."""
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


class AnthropicProvider:
    """An Anthropic Messages client with bounded concurrency and cost logging."""

    def __init__(self, *, api_key: str, max_concurrent: int, model: str = DEFAULT_MODEL) -> None:
        # max_retries=0: tenacity owns retries, so each one is logged and
        # cost-tracked; the SDK's built-in retry loop would be invisible.
        self._client = anthropic.AsyncAnthropic(api_key=api_key, max_retries=0)
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._model = model

    async def _send(
        self,
        *,
        messages: list[anthropic.types.MessageParam],
        system: str | None,
        max_tokens: int,
        operation: str,
        model: str | None,
        book_id: UUID | None,
        request_params: dict[str, Any],
    ) -> AnthropicResult:
        """Send one Messages request, retrying transient errors, logging each.

        `messages` is pre-built by the caller — `generate_text` sends plain
        string content, `generate_vision` sends an image block plus text.
        `request_params` is logged verbatim to `api_calls`, so callers keep
        bulky payloads (such as base64 image data) out of it.
        """
        use_model = model or self._model
        system_arg: str | anthropic.Omit = system if system is not None else anthropic.omit

        async with self._semaphore:
            async for attempt in make_async_retrying(_is_retryable, label="Anthropic"):
                with attempt:
                    retry_attempt = attempt.retry_state.attempt_number - 1
                    started = time.monotonic()
                    logger.debug(
                        "Anthropic {} model={} max_tokens={} attempt={}",
                        operation,
                        use_model,
                        max_tokens,
                        retry_attempt,
                    )
                    try:
                        message = await asyncio.wait_for(
                            self._client.messages.create(
                                model=use_model,
                                max_tokens=max_tokens,
                                system=system_arg,
                                messages=messages,
                            ),
                            timeout=_ATTEMPT_TIMEOUT_S,
                        )
                    except Exception as exc:
                        await record_api_call(
                            partial(
                                log_api_call,
                                provider=_PROVIDER,
                                endpoint=use_model,
                                operation=operation,
                                duration_ms=int((time.monotonic() - started) * 1000),
                                success=False,
                                book_id=book_id,
                                request_params=request_params,
                                cost_usd=Decimal(0),
                                error_type=type(exc).__name__,
                                error_message=str(exc),
                                retry_attempt=retry_attempt,
                            )
                        )
                        raise
                    duration_s = time.monotonic() - started
                    usage = message.usage
                    cost = cost_for_tokens(use_model, usage.input_tokens, usage.output_tokens)
                    text = _text_of(message)
                    await record_api_call(
                        partial(
                            log_api_call,
                            provider=_PROVIDER,
                            endpoint=use_model,
                            operation=operation,
                            duration_ms=int(duration_s * 1000),
                            success=True,
                            book_id=book_id,
                            request_params=request_params,
                            response_summary={
                                "input_tokens": usage.input_tokens,
                                "output_tokens": usage.output_tokens,
                                "stop_reason": message.stop_reason,
                                "text_chars": len(text),
                            },
                            cost_usd=cost,
                            retry_attempt=retry_attempt,
                        )
                    )
                    logger.debug(
                        "Anthropic {} ok in={} out={} cost=${} {:.1f}s",
                        operation,
                        usage.input_tokens,
                        usage.output_tokens,
                        cost,
                        duration_s,
                    )
                    return AnthropicResult(
                        text=text,
                        input_tokens=usage.input_tokens,
                        output_tokens=usage.output_tokens,
                        cost_usd=cost,
                        duration_s=duration_s,
                        stop_reason=message.stop_reason,
                    )
        raise AssertionError("unreachable: AsyncRetrying yielded no attempts")

    async def generate_text(
        self,
        *,
        prompt: str,
        system: str | None = None,
        max_tokens: int = 1024,
        operation: str = "generate_text",
        model: str | None = None,
        book_id: UUID | None = None,
    ) -> AnthropicResult:
        """Send one text-only Messages request."""
        messages: list[anthropic.types.MessageParam] = [{"role": "user", "content": prompt}]
        request_params: dict[str, Any] = {
            "model": model or self._model,
            "max_tokens": max_tokens,
            "system": system,
            "prompt": prompt,
        }
        return await self._send(
            messages=messages,
            system=system,
            max_tokens=max_tokens,
            operation=operation,
            model=model,
            book_id=book_id,
            request_params=request_params,
        )

    async def generate_vision(
        self,
        *,
        prompt: str,
        image_bytes: bytes,
        system: str | None = None,
        max_tokens: int = 1024,
        operation: str = "generate_vision",
        model: str | None = None,
        book_id: UUID | None = None,
    ) -> AnthropicResult:
        """Send one Messages request with a PNG image plus a text prompt.

        `image_bytes` must be PNG — the pipeline only ever evaluates the
        line-art PNGs it produces. The base64 payload is kept out of the
        logged `request_params`; only its byte length is recorded.
        """
        image_b64 = base64.standard_b64encode(image_bytes).decode("ascii")
        image_block: anthropic.types.ImageBlockParam = {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": image_b64},
        }
        text_block: anthropic.types.TextBlockParam = {"type": "text", "text": prompt}
        messages: list[anthropic.types.MessageParam] = [
            {"role": "user", "content": [image_block, text_block]}
        ]
        request_params: dict[str, Any] = {
            "model": model or self._model,
            "max_tokens": max_tokens,
            "system": system,
            "prompt": prompt,
            "image_png_bytes": len(image_bytes),
        }
        return await self._send(
            messages=messages,
            system=system,
            max_tokens=max_tokens,
            operation=operation,
            model=model,
            book_id=book_id,
            request_params=request_params,
        )

    async def generate_json(
        self,
        *,
        prompt: str,
        system: str | None = None,
        max_tokens: int = 1024,
        operation: str = "generate_json",
        model: str | None = None,
        book_id: UUID | None = None,
    ) -> tuple[dict[str, Any], AnthropicResult]:
        """Like `generate_text`, but require and parse a single JSON object."""
        full_system = _JSON_INSTRUCTION if system is None else f"{system}\n\n{_JSON_INSTRUCTION}"
        result = await self.generate_text(
            prompt=prompt,
            system=full_system,
            max_tokens=max_tokens,
            operation=operation,
            model=model,
            book_id=book_id,
        )
        try:
            parsed = json.loads(_strip_code_fence(result.text))
        except json.JSONDecodeError as exc:
            raise AnthropicProviderError(f"model response was not valid JSON: {exc}") from exc
        if not isinstance(parsed, dict):
            raise AnthropicProviderError(
                f"model JSON was a {type(parsed).__name__}, expected an object"
            )
        return parsed, result


def get_anthropic_provider() -> AnthropicProvider:
    """Build an `AnthropicProvider` from settings; raises if the key is unset."""
    settings = get_settings()
    if not settings.anthropic_api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set — add it to .env")
    return AnthropicProvider(
        api_key=settings.anthropic_api_key,
        max_concurrent=settings.anthropic_max_concurrent,
    )
