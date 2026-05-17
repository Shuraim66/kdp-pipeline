"""Deterministic image-prompt construction for Fal.ai generation."""

from __future__ import annotations

from src.config.schema import NicheConfig


def build_image_prompt(config: NicheConfig, subject: str, variation_idx: int) -> tuple[str, str]:
    """Build the ``(prompt, negative_prompt)`` pair for one image slot.

    Deterministic given its inputs. `variation_idx` selects a composition
    modifier, cycling through `composition_modifiers` so each variation of a
    subject is framed differently.
    """
    modifiers = config.composition_modifiers
    modifier = modifiers[variation_idx % len(modifiers)]
    prompt = " ".join(
        [
            config.style.art_style.strip(),
            subject,
            modifier,
            "isolated on pure white background",
            f"{config.style.line_weight} black lines",
            "coloring book page for adults",
            "vector style, professional illustration",
            "clean composition with margin around edges",
        ]
    )
    return prompt, config.style.negative_prompts
