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
    assert "coloring book page for adults" in prompt
    assert "isolated on pure white background" in prompt


def test_negative_prompt_is_the_style_value(niche_config) -> None:
    _, negative = build_image_prompt(niche_config, "anything", 3)
    assert negative == niche_config.style.negative_prompts


def test_modifier_cycles_through_variations(niche_config) -> None:
    modifiers = niche_config.composition_modifiers
    for variation_idx in range(len(modifiers) + 2):
        prompt, _ = build_image_prompt(niche_config, "a star", variation_idx)
        assert modifiers[variation_idx % len(modifiers)] in prompt
