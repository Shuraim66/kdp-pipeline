"""Tests for the deterministic image-prompt builder."""

from __future__ import annotations

from src.utils.prompts import build_image_prompt


def test_prompt_is_deterministic(niche_config) -> None:
    first = build_image_prompt(niche_config, "a teddy bear", 0)
    second = build_image_prompt(niche_config, "a teddy bear", 0)
    assert first == second


def test_prompt_contains_subject_and_style(niche_config) -> None:
    prompt, _ = build_image_prompt(niche_config, "a teddy bear", 0)
    assert "a teddy bear" in prompt
    assert niche_config.style.art_style.strip() in prompt
    assert niche_config.style.line_weight in prompt
    assert "simple black and white line drawing" in prompt
    assert "isolated on pure white background" in prompt


def test_prompt_front_loads_line_weight(niche_config) -> None:
    prompt, _ = build_image_prompt(niche_config, "a teddy bear", 0)
    assert prompt.startswith("thick black outlines")
    assert "thick continuous black lines 4-6 pixels wide" in prompt
    assert "in a cute simple children's cartoon style" in prompt


def test_prompt_avoids_text_triggering_words(niche_config) -> None:
    # schnell renders the literal word "book" as garbled title text — the
    # prompt and the niche's art_style must both keep it out.
    prompt, _ = build_image_prompt(niche_config, "a teddy bear", 0)
    assert "book" not in prompt.lower()


def test_prompt_enforces_connected_outlines(niche_config) -> None:
    prompt, _ = build_image_prompt(niche_config, "a teddy bear", 0)
    assert "fully connected outlines" in prompt
    assert "single coherent illustration" in prompt


def test_negative_prompt_strengthens_the_style_value(niche_config) -> None:
    _, negative = build_image_prompt(niche_config, "anything", 3)
    # The niche's own negatives are preserved, with line-quality terms appended.
    assert niche_config.style.negative_prompts.strip() in negative
    for term in (
        "thin lines",
        "hairline strokes",
        "broken lines",
        "disconnected pieces",
        "floating artifacts",
    ):
        assert term in negative


def test_modifier_cycles_through_variations(niche_config) -> None:
    modifiers = niche_config.composition_modifiers
    for variation_idx in range(len(modifiers) + 2):
        prompt, _ = build_image_prompt(niche_config, "a star", variation_idx)
        assert modifiers[variation_idx % len(modifiers)] in prompt
