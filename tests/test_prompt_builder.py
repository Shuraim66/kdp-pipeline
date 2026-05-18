"""Tests for the deterministic image-prompt builder."""

from __future__ import annotations

from typing import Literal

from src.config.schema import Subject
from src.utils.prompts import build_image_prompt


def _subj(name: str, kind: Literal["object", "character", "scene"] = "object") -> Subject:
    """A Subject for tests — defaults to the `object` kind."""
    return Subject(name=name, kind=kind)


def test_prompt_is_deterministic(niche_config) -> None:
    first = build_image_prompt(niche_config, _subj("a teddy bear"), 0)
    second = build_image_prompt(niche_config, _subj("a teddy bear"), 0)
    assert first == second


def test_prompt_contains_subject_and_style(niche_config) -> None:
    prompt, _ = build_image_prompt(niche_config, _subj("a teddy bear"), 0)
    assert "a teddy bear" in prompt
    assert niche_config.style.art_style.strip() in prompt
    assert niche_config.style.line_weight in prompt
    assert "on a plain white background" in prompt


def test_prompt_front_loads_subject(make_niche_config) -> None:
    # With no LoRA the subject leads — FLUX dev drew blank pages when the
    # prompt opened with a wall of style wording.
    prompt, _ = build_image_prompt(make_niche_config(), _subj("a teddy bear"), 0)
    assert prompt.startswith("a teddy bear")
    assert "thick continuous black lines 4-6 pixels wide" in prompt
    assert "in a cute simple children's cartoon style" in prompt


def test_prompt_leads_with_lora_trigger(make_niche_config) -> None:
    # A LoRA's trigger phrase must lead the prompt to activate its style.
    config = make_niche_config(
        loras=[{"path": "https://example.com/cb.safetensors", "trigger": "c0l0r book"}]
    )
    prompt, _ = build_image_prompt(config, _subj("a teddy bear"), 0)
    assert prompt.startswith("c0l0r book, a teddy bear")


def test_prompt_mentions_white_background_once(niche_config) -> None:
    # Repeated "white background" saturated the prompt and blanked dev output.
    prompt, _ = build_image_prompt(niche_config, _subj("a teddy bear"), 0)
    assert prompt.lower().count("white background") == 1


def test_prompt_avoids_text_triggering_words(make_niche_config) -> None:
    # Without a LoRA the prompt carries no "book" — the bare model renders the
    # literal word as garbled title text. (A LoRA trigger may legitimately
    # contain it; the LoRA is trained to read it as a style cue, not text.)
    prompt, _ = build_image_prompt(make_niche_config(), _subj("a teddy bear"), 0)
    assert "book" not in prompt.lower()


def test_prompt_enforces_connected_outlines(niche_config) -> None:
    prompt, _ = build_image_prompt(niche_config, _subj("a teddy bear"), 0)
    assert "fully connected outlines" in prompt
    assert "single coherent illustration" in prompt


def test_prompt_dispatches_on_subject_kind(make_niche_config) -> None:
    # Each `kind` appends its own framing directive — the Q4 fix that stops the
    # LoRA wrapping a bare object in an invented character.
    config = make_niche_config()
    obj, _ = build_image_prompt(config, _subj("a stethoscope", "object"), 0)
    char, _ = build_image_prompt(config, _subj("a teddy bear", "character"), 0)
    scene, _ = build_image_prompt(config, _subj("a nurse at a bedside", "scene"), 0)
    assert "no added characters, animals, or mascots" in obj
    assert "single character centered on the page" in char
    assert "the elements in the scene belong together naturally" in scene


def test_negative_prompt_strengthens_the_style_value(niche_config) -> None:
    _, negative = build_image_prompt(niche_config, _subj("anything"), 3)
    # The niche's own negatives are preserved, with line-quality terms appended.
    assert niche_config.style.negative_prompts.strip() in negative
    for term in (
        "thin lines",
        "hairline strokes",
        "broken lines",
        "disconnected pieces",
        "floating artifacts",
        "scattered dots",
    ):
        assert term in negative


def test_modifier_cycles_through_variations(niche_config) -> None:
    modifiers = niche_config.composition_modifiers
    for variation_idx in range(len(modifiers) + 2):
        prompt, _ = build_image_prompt(niche_config, _subj("a star"), variation_idx)
        assert modifiers[variation_idx % len(modifiers)] in prompt
