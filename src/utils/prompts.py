"""Deterministic image-prompt construction for Fal.ai generation."""

from __future__ import annotations

from src.config.schema import NicheConfig

# Universal line-quality negatives, appended to every niche's own negative
# prompt. They target the diffusion model's observed failure modes — thin,
# sketchy, broken strokes — independent of subject.
#
# Note: the plain fal-ai/flux/* endpoints take no negative prompt, so this is
# unused until the pipeline moves to the fal-ai/flux-lora endpoint (Q2), which
# does accept negative_prompt.
_LINE_QUALITY_NEGATIVES = (
    "thin lines, hairline strokes, sketchy, pencil sketch, broken lines, "
    "gaps in lines, disconnected pieces, incomplete shapes, floating artifacts"
)


def build_image_prompt(config: NicheConfig, subject: str, variation_idx: int) -> tuple[str, str]:
    """Build the ``(prompt, negative_prompt)`` pair for one image slot.

    Deterministic given its inputs. `variation_idx` selects a composition
    modifier, cycling through `composition_modifiers` so each variation of a
    subject is framed differently.

    Any configured LoRA trigger phrases lead the prompt — a LoRA needs its
    trigger to activate its trained style. The subject comes next: diffusion
    models weight leading tokens heavily, and FLUX dev drew blank pages when
    the prompt opened with a wall of style wording. Negations ("no colour",
    "no shading") live only in the negative prompt, and "white background"
    appears once, near the end.
    """
    modifiers = config.composition_modifiers
    modifier = modifiers[variation_idx % len(modifiers)]
    triggers = [lora.trigger for lora in config.generation.loras if lora.trigger]
    prompt = ", ".join(
        [
            *triggers,
            subject,
            "a single coherent illustration, one complete object with fully connected outlines",
            config.style.art_style.strip(),
            "in a cute simple children's cartoon style",
            f"{config.style.line_weight} black outlines",
            "thick continuous black lines 4-6 pixels wide",
            modifier,
            "centered with a comfortable margin around the edges",
            "on a plain white background",
        ]
    )
    negative_prompt = f"{config.style.negative_prompts.strip()}, {_LINE_QUALITY_NEGATIVES}"
    return prompt, negative_prompt
