"""Fal.ai image-generation client — async, retried, cost-tracked.

`FalProvider.generate_image` is a coroutine called (via `asyncio.run`) from
the otherwise-synchronous pipeline. It bounds concurrency with a semaphore,
retries transient failures with the shared backoff policy, applies a hard
per-attempt timeout, downloads the rendered image bytes, and records every
attempt — success or failure — to the `api_calls` table.
"""

from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass
from decimal import Decimal
from functools import partial
from typing import Any
from uuid import UUID

import httpx
from fal_client import AsyncClient, FalClientHTTPError

from src.db.repos.api_calls import log_api_call
from src.providers._common import record_api_call
from src.settings import get_settings
from src.utils.logging import logger
from src.utils.retries import make_async_retrying

_PROVIDER = "fal"
_OPERATION = "generate_image"

# Fal.ai pricing, verified against the model pages on 2026-05-18: every FLUX
# endpoint bills per megapixel, rounding the image area UP to the next whole
# megapixel. A 2048x2048 image is 5 MP — e.g. flux/dev costs $0.125. Revisit
# if Fal changes its rates.
_RATE_PER_MEGAPIXEL: dict[str, Decimal] = {
    "fal-ai/flux/schnell": Decimal("0.003"),
    "fal-ai/flux/dev": Decimal("0.025"),
    "fal-ai/flux-lora": Decimal("0.035"),
}
# Fallback for an unrecognised model: the priciest known rate, so an unknown
# model over-reports rather than under-reports spend.
_FALLBACK_RATE_PER_MEGAPIXEL = Decimal("0.025")

# Hard ceiling on one generation attempt (queue + render + download).
_ATTEMPT_TIMEOUT_S = 60.0
# Tighter ceiling on just the result-image download.
_DOWNLOAD_TIMEOUT_S = 30.0


class FalProviderError(RuntimeError):
    """A Fal response was structurally unusable (e.g. it carried no image)."""


@dataclass(frozen=True, slots=True)
class FalImageResult:
    """The outcome of one successful `generate_image` call."""

    image_bytes: bytes
    seed: int
    width: int
    height: int
    duration_s: float
    cost_usd: Decimal


def _megapixels(width: int, height: int) -> int:
    """Image area in megapixels, rounded up — Fal's billing unit."""
    return math.ceil(width * height / 1_000_000)


def cost_for_image(model: str, width: int, height: int) -> Decimal:
    """Fal's charge for one rendered image of the given model and size."""
    rate = _RATE_PER_MEGAPIXEL.get(model)
    if rate is None:
        logger.warning("unknown Fal model {!r}; costing at fallback rate", model)
        rate = _FALLBACK_RATE_PER_MEGAPIXEL
    return (rate * _megapixels(width, height)).quantize(Decimal("0.00001"))


def _is_retryable(exc: BaseException) -> bool:
    """Transient Fal/transport faults worth retrying; auth and 4xx are not."""
    if isinstance(exc, FalClientHTTPError):
        return exc.status_code == 429 or exc.status_code >= 500
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code >= 500
    # TimeoutError also covers asyncio's wait_for timeout and FalClientTimeoutError.
    return isinstance(
        exc,
        httpx.ConnectError
        | httpx.ConnectTimeout
        | httpx.ReadTimeout
        | httpx.RemoteProtocolError
        | TimeoutError,
    )


class FalProvider:
    """A Fal.ai client with bounded concurrency, retries, and cost logging."""

    def __init__(self, *, api_key: str, max_concurrent: int) -> None:
        self._client = AsyncClient(key=api_key)
        self._semaphore = asyncio.Semaphore(max_concurrent)

    async def generate_image(
        self,
        *,
        prompt: str,
        model: str,
        width: int,
        height: int,
        num_inference_steps: int,
        seed: int | None = None,
        guidance_scale: float | None = None,
        negative_prompt: str | None = None,
        loras: list[dict[str, Any]] | None = None,
        book_id: UUID | None = None,
        image_id: UUID | None = None,
    ) -> FalImageResult:
        """Render one image, retrying transient errors, logging every attempt.

        `negative_prompt` and `loras` are only accepted by the
        `fal-ai/flux-lora` endpoint; pass them only when that model is in use.
        """
        arguments: dict[str, Any] = {
            "prompt": prompt,
            "image_size": {"width": width, "height": height},
            "num_inference_steps": num_inference_steps,
            "num_images": 1,
            # Coloring-book line art trips no policy; a tripped checker just
            # returns a black image and wastes a QA cycle, so disable it.
            "enable_safety_checker": False,
            "output_format": "png",
        }
        if seed is not None:
            arguments["seed"] = seed
        if guidance_scale is not None:
            arguments["guidance_scale"] = guidance_scale
        if negative_prompt is not None:
            arguments["negative_prompt"] = negative_prompt
        if loras:
            arguments["loras"] = loras
        request_params: dict[str, Any] = {"model": model, **arguments}

        async with self._semaphore:
            async for attempt in make_async_retrying(_is_retryable, label="Fal"):
                with attempt:
                    retry_attempt = attempt.retry_state.attempt_number - 1
                    started = time.monotonic()
                    logger.debug(
                        "Fal {} model={} size={}x{} steps={} seed={} attempt={}",
                        _OPERATION,
                        model,
                        width,
                        height,
                        num_inference_steps,
                        seed,
                        retry_attempt,
                    )
                    try:
                        image_bytes, used_seed, out_w, out_h = await asyncio.wait_for(
                            self._generate_once(model, arguments),
                            timeout=_ATTEMPT_TIMEOUT_S,
                        )
                    except Exception as exc:
                        await record_api_call(
                            partial(
                                log_api_call,
                                provider=_PROVIDER,
                                endpoint=model,
                                operation=_OPERATION,
                                duration_ms=int((time.monotonic() - started) * 1000),
                                success=False,
                                book_id=book_id,
                                image_id=image_id,
                                request_params=request_params,
                                cost_usd=Decimal(0),
                                error_type=type(exc).__name__,
                                error_message=str(exc),
                                retry_attempt=retry_attempt,
                            )
                        )
                        raise
                    duration_s = time.monotonic() - started
                    cost = cost_for_image(model, out_w, out_h)
                    await record_api_call(
                        partial(
                            log_api_call,
                            provider=_PROVIDER,
                            endpoint=model,
                            operation=_OPERATION,
                            duration_ms=int(duration_s * 1000),
                            success=True,
                            book_id=book_id,
                            image_id=image_id,
                            request_params=request_params,
                            response_summary={
                                "seed": used_seed,
                                "width": out_w,
                                "height": out_h,
                                "bytes": len(image_bytes),
                                "megapixels": _megapixels(out_w, out_h),
                            },
                            cost_usd=cost,
                            retry_attempt=retry_attempt,
                        )
                    )
                    logger.debug(
                        "Fal {} ok seed={} {}x{} {:.1f}s cost=${}",
                        _OPERATION,
                        used_seed,
                        out_w,
                        out_h,
                        duration_s,
                        cost,
                    )
                    return FalImageResult(
                        image_bytes=image_bytes,
                        seed=used_seed,
                        width=out_w,
                        height=out_h,
                        duration_s=duration_s,
                        cost_usd=cost,
                    )
        raise AssertionError("unreachable: AsyncRetrying yielded no attempts")

    async def _generate_once(
        self, model: str, arguments: dict[str, Any]
    ) -> tuple[bytes, int, int, int]:
        """One Fal generation: submit, await the result, download image bytes."""
        result = await self._client.subscribe(model, arguments=arguments)
        if not isinstance(result, dict):
            raise FalProviderError(f"Fal returned a non-object result: {type(result)!r}")
        images = result.get("images") or []
        if not images:
            raise FalProviderError("Fal response contained no images")
        image = images[0]
        url = image.get("url")
        if not url:
            raise FalProviderError("Fal image entry carried no URL")
        size = arguments["image_size"]
        width = int(image.get("width") or size["width"])
        height = int(image.get("height") or size["height"])
        seed = int(result.get("seed") or arguments.get("seed") or 0)
        image_bytes = await self._download(url)
        logger.debug("Fal downloaded {} bytes from result image", len(image_bytes))
        return image_bytes, seed, width, height

    @staticmethod
    async def _download(url: str) -> bytes:
        """Fetch the rendered image bytes from a Fal result URL."""
        async with httpx.AsyncClient(timeout=_DOWNLOAD_TIMEOUT_S) as client:
            response = await client.get(url)
            response.raise_for_status()
            return response.content


def get_fal_provider() -> FalProvider:
    """Build a `FalProvider` from settings; raises if `FAL_KEY` is unset."""
    settings = get_settings()
    if not settings.fal_key:
        raise RuntimeError("FAL_KEY is not set — add it to .env before generating images")
    return FalProvider(
        api_key=settings.fal_key,
        max_concurrent=settings.fal_max_concurrent,
    )
