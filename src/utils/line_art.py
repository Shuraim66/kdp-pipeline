"""Line-art normalisation — clean, thicken, and upscale generated pages.

The diffusion model emits thin, antialiased strokes at its own (capped)
resolution. `normalize_line_art` turns each page into crisp, uniformly bold
black line art at the print resolution QA expects — a deterministic step that
fixes line weight, kills stray grey, and clears the 300-DPI requirement
without another API call.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageFilter

# A grayscale pixel darker than this (0-255) is treated as ink.
_INK_THRESHOLD = 200
# MinFilter window for stroke dilation — 5 turns the model's thin strokes bold.
_DILATE_WINDOW = 5
# Re-binarise cutoff applied after the upscale interpolation.
_UPSCALE_CUTOFF = 128


def _binarise(image: Image.Image, threshold: int) -> Image.Image:
    """Flatten to pure black-on-white, dropping antialiasing and any shading."""
    array = np.asarray(image.convert("L"))
    binary = np.where(array < threshold, 0, 255).astype(np.uint8)
    return Image.fromarray(binary, mode="L")


def normalize_line_art(image_bytes: bytes, *, target_size: tuple[int, int]) -> bytes:
    """Clean, thicken, and upscale one generated coloring-book page.

    Three deterministic steps:

      1. binarise — kill antialiasing and any faint shading the model added;
      2. dilate the ink (`MinFilter`) to a consistent bold stroke weight;
      3. upscale to `target_size`, then re-binarise so the interpolation
         leaves no grey halo.

    Returns PNG bytes of an ``L``-mode pure black-and-white image sized exactly
    `target_size`.
    """
    with Image.open(io.BytesIO(image_bytes)) as handle:
        cleaned = _binarise(handle, _INK_THRESHOLD)
    bold = cleaned.filter(ImageFilter.MinFilter(_DILATE_WINDOW))
    upscaled = bold.resize(target_size, Image.Resampling.LANCZOS)
    final = _binarise(upscaled, _UPSCALE_CUTOFF)
    buffer = io.BytesIO()
    final.save(buffer, format="PNG")
    return buffer.getvalue()
