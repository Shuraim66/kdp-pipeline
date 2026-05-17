"""Tests for the Fal.ai and Anthropic provider clients.

Unit tests mock the network seam (`FalProvider._generate_once`,
`AsyncMessages.create`) so they run offline and fast. The two integration
tests make one real API call each; they are marked `integration` and skip
unless the corresponding key is configured.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock

import anthropic
import httpx
import pytest
from fal_client import FalClientHTTPError
from src.providers.anthropic import (
    AnthropicProvider,
    AnthropicProviderError,
    cost_for_tokens,
)
from src.providers.fal import FalImageResult, FalProvider, cost_for_image
from src.settings import Settings, get_settings
from src.utils.retries import make_async_retrying

_PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32


def _noop_log(**_kwargs: Any) -> None:
    """Stand-in for `log_api_call` — records nothing, touches no database."""


async def _no_sleep(_seconds: float) -> None:
    """A retry sleep that returns immediately, so retry tests do not wait."""


def _instant_retrying(is_retryable: Any, **kwargs: Any) -> Any:
    """`make_async_retrying` with the backoff sleep removed (for tests)."""
    return make_async_retrying(is_retryable, sleep=_no_sleep, **kwargs)


def _http_error(status: int) -> FalClientHTTPError:
    return FalClientHTTPError(
        f"HTTP {status}",
        status_code=status,
        response_headers={},
        response=httpx.Response(status),
    )


def _api_status_error(status: int) -> anthropic.APIStatusError:
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(status, request=request)
    return anthropic.APIStatusError(f"HTTP {status}", response=response, body=None)


def _fake_message(
    text: str, in_tok: int, out_tok: int, stop: str = "end_turn"
) -> anthropic.types.Message:
    block = anthropic.types.TextBlock.model_construct(type="text", text=text)
    usage = anthropic.types.Usage.model_construct(input_tokens=in_tok, output_tokens=out_tok)
    return anthropic.types.Message.model_construct(content=[block], usage=usage, stop_reason=stop)


# --- Fal: cost ---------------------------------------------------------------


def test_cost_for_image_schnell_rounds_megapixels_up() -> None:
    # 2550x2550 = 6.50 MP -> billed as 7 MP at $0.003/MP.
    assert cost_for_image("fal-ai/flux/schnell", 2550, 2550) == Decimal("0.021")


def test_cost_for_image_dev_rate() -> None:
    # 1000x1000 = exactly 1 MP at the $0.025/MP dev rate.
    assert cost_for_image("fal-ai/flux/dev", 1000, 1000) == Decimal("0.025")


def test_cost_for_image_unknown_model_uses_fallback() -> None:
    # Unknown model falls back to the priciest known rate ($0.025/MP).
    assert cost_for_image("fal-ai/flux/pro", 1000, 1000) == Decimal("0.025")


# --- Fal: generate_image -----------------------------------------------------


async def test_fal_generate_image_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("src.providers.fal.log_api_call", _noop_log)
    provider = FalProvider(api_key="test", max_concurrent=2)
    once = AsyncMock(return_value=(_PNG, 4242, 2550, 2550))
    monkeypatch.setattr(provider, "_generate_once", once)

    result = await provider.generate_image(
        prompt="a bold-line star",
        model="fal-ai/flux/schnell",
        width=2550,
        height=2550,
        num_inference_steps=4,
        seed=4242,
    )

    assert isinstance(result, FalImageResult)
    assert result.image_bytes == _PNG
    assert result.seed == 4242
    assert result.width == result.height == 2550
    assert result.cost_usd == Decimal("0.021")
    once.assert_awaited_once()


async def test_fal_generate_image_retries_then_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logged: list[dict[str, Any]] = []
    monkeypatch.setattr("src.providers.fal.log_api_call", lambda **kw: logged.append(kw))
    monkeypatch.setattr("src.providers.fal.make_async_retrying", _instant_retrying)
    provider = FalProvider(api_key="test", max_concurrent=1)
    once = AsyncMock(side_effect=[_http_error(503), (_PNG, 7, 1000, 1000)])
    monkeypatch.setattr(provider, "_generate_once", once)

    result = await provider.generate_image(
        prompt="x",
        model="fal-ai/flux/schnell",
        width=1000,
        height=1000,
        num_inference_steps=4,
    )

    assert result.seed == 7
    assert once.await_count == 2
    # Two api_calls rows: the failed attempt (cost 0), then the success.
    assert [r["success"] for r in logged] == [False, True]
    assert [r["retry_attempt"] for r in logged] == [0, 1]
    assert logged[0]["cost_usd"] == Decimal(0)
    assert logged[1]["cost_usd"] == Decimal("0.003")


async def test_fal_generate_image_auth_error_not_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logged: list[dict[str, Any]] = []
    monkeypatch.setattr("src.providers.fal.log_api_call", lambda **kw: logged.append(kw))
    monkeypatch.setattr("src.providers.fal.make_async_retrying", _instant_retrying)
    provider = FalProvider(api_key="bad", max_concurrent=1)
    once = AsyncMock(side_effect=_http_error(401))
    monkeypatch.setattr(provider, "_generate_once", once)

    with pytest.raises(FalClientHTTPError):
        await provider.generate_image(
            prompt="x",
            model="fal-ai/flux/schnell",
            width=1000,
            height=1000,
            num_inference_steps=4,
        )
    # A 401 is fatal: one attempt, one failure row, no retry.
    assert once.await_count == 1
    assert [r["success"] for r in logged] == [False]


# --- Anthropic: cost ---------------------------------------------------------


def test_cost_for_tokens_sonnet() -> None:
    # 1000 in @ $3/Mtok + 500 out @ $15/Mtok = 0.003 + 0.0075.
    assert cost_for_tokens("claude-sonnet-4-6", 1000, 500) == Decimal("0.01050")


def test_cost_for_tokens_unknown_model_is_zero() -> None:
    assert cost_for_tokens("gpt-fictional", 1000, 500) == Decimal(0)


# --- Anthropic: generate_text / generate_json --------------------------------


async def test_anthropic_generate_text_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("src.providers.anthropic.log_api_call", _noop_log)
    provider = AnthropicProvider(api_key="test", max_concurrent=1)
    create = AsyncMock(return_value=_fake_message("hello there", 1000, 200))
    monkeypatch.setattr(provider._client.messages, "create", create)

    result = await provider.generate_text(prompt="hi", max_tokens=64)

    assert result.text == "hello there"
    assert result.input_tokens == 1000
    assert result.output_tokens == 200
    assert result.cost_usd == cost_for_tokens("claude-sonnet-4-6", 1000, 200)
    assert result.stop_reason == "end_turn"
    create.assert_awaited_once()


async def test_anthropic_generate_json_strips_code_fence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("src.providers.anthropic.log_api_call", _noop_log)
    provider = AnthropicProvider(api_key="test", max_concurrent=1)
    fenced = '```json\n{"title": "Calm Nurses", "pages": 50}\n```'
    create = AsyncMock(return_value=_fake_message(fenced, 800, 60))
    monkeypatch.setattr(provider._client.messages, "create", create)

    data, result = await provider.generate_json(prompt="make metadata")

    assert data == {"title": "Calm Nurses", "pages": 50}
    assert result.output_tokens == 60


async def test_anthropic_generate_json_rejects_non_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("src.providers.anthropic.log_api_call", _noop_log)
    provider = AnthropicProvider(api_key="test", max_concurrent=1)
    create = AsyncMock(return_value=_fake_message("sorry, I cannot", 10, 10))
    monkeypatch.setattr(provider._client.messages, "create", create)

    with pytest.raises(AnthropicProviderError):
        await provider.generate_json(prompt="make metadata")


async def test_anthropic_generate_text_retries_on_overload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logged: list[dict[str, Any]] = []
    monkeypatch.setattr("src.providers.anthropic.log_api_call", lambda **kw: logged.append(kw))
    monkeypatch.setattr("src.providers.anthropic.make_async_retrying", _instant_retrying)
    provider = AnthropicProvider(api_key="test", max_concurrent=1)
    create = AsyncMock(side_effect=[_api_status_error(529), _fake_message("ok", 100, 20)])
    monkeypatch.setattr(provider._client.messages, "create", create)

    result = await provider.generate_text(prompt="hi")

    assert result.text == "ok"
    assert create.await_count == 2
    assert [r["success"] for r in logged] == [False, True]


async def test_anthropic_generate_text_bad_request_not_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logged: list[dict[str, Any]] = []
    monkeypatch.setattr("src.providers.anthropic.log_api_call", lambda **kw: logged.append(kw))
    monkeypatch.setattr("src.providers.anthropic.make_async_retrying", _instant_retrying)
    provider = AnthropicProvider(api_key="test", max_concurrent=1)
    create = AsyncMock(side_effect=_api_status_error(400))
    monkeypatch.setattr(provider._client.messages, "create", create)

    with pytest.raises(anthropic.APIStatusError):
        await provider.generate_text(prompt="hi")
    # A 400 is fatal: one attempt, one failure row, no retry.
    assert create.await_count == 1
    assert [r["success"] for r in logged] == [False]


# --- Integration: one live call per provider ---------------------------------


def _settings_or_skip() -> Settings:
    try:
        return get_settings()
    except Exception as exc:  # any settings failure -> skip the live test
        pytest.skip(f"settings unavailable: {exc}")


@pytest.mark.integration
async def test_fal_generate_image_live(monkeypatch: pytest.MonkeyPatch) -> None:
    """One real Fal render — skipped unless FAL_KEY is configured."""
    settings = _settings_or_skip()
    if not settings.fal_key:
        pytest.skip("FAL_KEY not configured")
    monkeypatch.setattr("src.providers.fal.log_api_call", _noop_log)
    provider = FalProvider(api_key=settings.fal_key, max_concurrent=1)

    result = await provider.generate_image(
        prompt="bold simple line art of a single star, coloring book page, "
        "isolated on pure white background",
        model="fal-ai/flux/schnell",
        width=1024,
        height=1024,
        num_inference_steps=4,
    )
    assert result.image_bytes[:8] == b"\x89PNG\r\n\x1a\n"
    assert result.cost_usd > 0


@pytest.mark.integration
async def test_anthropic_generate_json_live(monkeypatch: pytest.MonkeyPatch) -> None:
    """One real Anthropic call — skipped unless ANTHROPIC_API_KEY is set."""
    settings = _settings_or_skip()
    if not settings.anthropic_api_key:
        pytest.skip("ANTHROPIC_API_KEY not configured")
    monkeypatch.setattr("src.providers.anthropic.log_api_call", _noop_log)
    provider = AnthropicProvider(api_key=settings.anthropic_api_key, max_concurrent=1)

    data, result = await provider.generate_json(
        prompt='Return a JSON object with exactly one key "ok" whose value is the boolean true.',
        max_tokens=64,
    )
    assert data.get("ok") is True
    assert result.input_tokens > 0
    assert result.cost_usd > 0
