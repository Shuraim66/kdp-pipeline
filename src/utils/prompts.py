"""Deterministic image-prompt construction for Fal.ai generation."""

from __future__ import annotations

from src.config.schema import NicheConfig, Subject

# Universal line-quality negatives, appended to every niche's own negative
# prompt. They target the diffusion model's observed failure modes — thin,
# sketchy, broken strokes, and the scattered speckle/stray-dot noise Vision QA
# flagged in Q3 — independent of subject.
#
# Note: the plain fal-ai/flux/* endpoints take no negative prompt, so this is
# unused until the pipeline moves to the fal-ai/flux-lora endpoint (Q2), which
# does accept negative_prompt.
_LINE_QUALITY_NEGATIVES = (
    "thin lines, hairline strokes, sketchy, pencil sketch, broken lines, "
    "gaps in lines, disconnected pieces, incomplete shapes, floating artifacts, "
    "scattered dots, stray marks, random speckles, decorative noise"
)

# Per-`kind` framing directive — keeps the LoRA from inventing a wrapper
# character around an object (Q3: "a cute stethoscope" came back as a cat
# holding a stethoscope), and tells a scene to stay coherent.
_KIND_DIRECTIVE: dict[str, str] = {
    "object": (
        "the object alone, isolated illustration, no added characters, "
        "animals, or mascots, no scene or environment"
    ),
    "character": (
        "single character centered on the page, no additional characters "
        "or creatures, simple background or none"
    ),
    "scene": "the elements in the scene belong together naturally",
}


def build_image_prompt(
    config: NicheConfig, subject: Subject, variation_idx: int
) -> tuple[str, str]:
    """Build the ``(prompt, negative_prompt)`` pair for one image slot.

    Deterministic given its inputs. `variation_idx` selects a composition
    modifier, cycling through `composition_modifiers` so each variation of a
    subject is framed differently.

    Any configured LoRA trigger phrases lead the prompt — a LoRA needs its
    trigger to activate its trained style. The subject comes next: diffusion
    models weight leading tokens heavily, and FLUX dev drew blank pages when
    the prompt opened with a wall of style wording. A `kind`-specific framing
    directive follows, so an object is drawn isolated rather than handed to an
    invented character. Negations live only in the negative prompt, and "white
    background" appears once, near the end.
    """
    coloring = config.require_coloring()
    modifiers = coloring.composition_modifiers
    modifier = modifiers[variation_idx % len(modifiers)]
    triggers = [lora.trigger for lora in coloring.generation.loras if lora.trigger]
    prompt = ", ".join(
        [
            *triggers,
            subject.name,
            _KIND_DIRECTIVE[subject.kind],
            "a single coherent illustration with fully connected outlines",
            coloring.style.art_style.strip(),
            "in a cute simple children's cartoon style",
            f"{coloring.style.line_weight} black outlines",
            "thick continuous black lines 4-6 pixels wide, every line the same "
            "uniform bold weight including interior detail lines",
            modifier,
            "on a plain white background",
        ]
    )
    negative_prompt = f"{coloring.style.negative_prompts.strip()}, {_LINE_QUALITY_NEGATIVES}"
    return prompt, negative_prompt
