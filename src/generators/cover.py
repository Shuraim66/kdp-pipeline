"""Cover assembly — hero illustration, composited wrap cover, PDF variants.

The cover is one rasterised image (back | spine | front, with bleed) at print
DPI; text is drawn into the pixels with Pillow, so the exported PDF embeds a
single image and carries no fonts. Three layout variants are produced for the
user to choose between.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

from src.config.schema import NicheConfig
from src.db.models import Book
from src.providers.fal import FalProvider
from src.utils.fonts import CoverFonts, load_cover_fonts
from src.utils.kdp_specs import (
    POINTS_PER_INCH,
    SAFE_MARGIN_IN,
    CoverDimensions,
    compute_cover_dimensions,
    parse_trim_size,
)
from src.utils.logging import logger

VARIANTS = ("a", "b", "c")

# Hero illustration — Fal.ai FLUX dev for higher quality (~$0.025/image).
_HERO_MODEL = "fal-ai/flux/dev"
_HERO_PX = 1500
_HERO_STEPS = 28
_HERO_GUIDANCE = 3.5

# KDP forbids spine text below this spine width.
_MIN_SPINE_TEXT_IN = 0.25

_RGB = tuple[int, int, int]


@dataclass(frozen=True, slots=True)
class CoverLayout:
    """Pixel geometry of a wrap cover — back, spine, and front columns."""

    dims: CoverDimensions
    dpi: int
    width: int
    height: int
    bleed_px: int
    safe_px: int
    back: tuple[int, int]
    spine: tuple[int, int]
    front: tuple[int, int]


def cover_layout(
    interior_page_count: int,
    trim_w_in: float,
    trim_h_in: float,
    *,
    paper: str = "white",
    dpi: int = 300,
) -> CoverLayout:
    """Compute the pixel geometry of the full wrap cover."""
    dims = compute_cover_dimensions(interior_page_count, trim_w_in, trim_h_in, paper=paper, dpi=dpi)
    bleed_px = round(dims.bleed_in * dpi)
    trim_w_px = round(trim_w_in * dpi)
    spine_px = round(dims.spine_width_in * dpi)
    back0 = bleed_px
    back1 = back0 + trim_w_px
    spine1 = back1 + spine_px
    front1 = spine1 + trim_w_px
    return CoverLayout(
        dims=dims,
        dpi=dpi,
        width=dims.total_width_px,
        height=dims.total_height_px,
        bleed_px=bleed_px,
        safe_px=round(SAFE_MARGIN_IN * dpi),
        back=(back0, back1),
        spine=(back1, spine1),
        front=(spine1, front1),
    )


def _hex_rgb(value: str) -> _RGB:
    text = value.lstrip("#")
    return int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16)


async def generate_hero(provider: FalProvider, config: NicheConfig, *, output_path: Path) -> Path:
    """Generate the front-cover hero illustration via Fal.ai FLUX dev."""
    prompt = (
        f"{config.cover.hero_subject}, {config.cover.hero_style}, "
        "isolated on a solid color background, no text, no letters, no numbers"
    )
    result = await provider.generate_image(
        prompt=prompt,
        model=_HERO_MODEL,
        width=_HERO_PX,
        height=_HERO_PX,
        num_inference_steps=_HERO_STEPS,
        guidance_scale=_HERO_GUIDANCE,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(result.image_bytes)
    logger.info("cover hero illustration saved: {}", output_path)
    return output_path


# --- text helpers ------------------------------------------------------------


def _font(path: Path, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(path), max(8, size))


def _wrap(
    draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, max_width: float
) -> list[str]:
    lines: list[str] = []
    current = ""
    for word in text.split():
        candidate = f"{current} {word}".strip()
        if current and draw.textlength(candidate, font=font) > max_width:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines or [""]


def _fit_font(
    draw: ImageDraw.ImageDraw,
    path: Path,
    text: str,
    *,
    max_width: float,
    max_lines: int,
    max_size: int,
) -> ImageFont.FreeTypeFont:
    """Largest font (<= max_size) at which `text` wraps within the line budget."""
    size = max_size
    while size > 24:
        font = _font(path, size)
        if len(_wrap(draw, text, font, max_width)) <= max_lines:
            return font
        size -= 4
    return _font(path, 24)


def _draw_wrapped(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont,
    *,
    center_x: float,
    top_y: float,
    max_width: float,
    fill: _RGB,
    spacing: float = 1.12,
) -> float:
    """Draw horizontally centred, wrapped text; return the y below the block."""
    ascent, descent = font.getmetrics()
    line_height = (ascent + descent) * spacing
    y = top_y
    for line in _wrap(draw, text, font, max_width):
        width = draw.textlength(line, font=font)
        draw.text((center_x - width / 2, y), line, font=font, fill=fill)
        y += line_height
    return y


def _draw_centered(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont,
    *,
    center_x: float,
    y: float,
    fill: _RGB,
) -> None:
    width = draw.textlength(text, font=font)
    draw.text((center_x - width / 2, y), text, font=font, fill=fill)


# --- hero placement ----------------------------------------------------------


def _paste_hero_fit(
    canvas_img: Image.Image, hero: Image.Image, box: tuple[int, int, int, int]
) -> None:
    """Scale the hero to fit inside `box`, preserving aspect, centred."""
    x0, y0, x1, y1 = box
    box_w, box_h = x1 - x0, y1 - y0
    if box_w <= 0 or box_h <= 0:
        return
    scale = min(box_w / hero.width, box_h / hero.height)
    size = (max(1, round(hero.width * scale)), max(1, round(hero.height * scale)))
    resized = hero.resize(size, Image.Resampling.LANCZOS)
    pos = (x0 + (box_w - size[0]) // 2, y0 + (box_h - size[1]) // 2)
    canvas_img.paste(resized, pos, resized if resized.mode == "RGBA" else None)


def _paste_hero_fill(
    canvas_img: Image.Image, hero: Image.Image, box: tuple[int, int, int, int]
) -> None:
    """Scale the hero to cover `box`, cropping the overflow."""
    x0, y0, x1, y1 = box
    box_w, box_h = x1 - x0, y1 - y0
    if box_w <= 0 or box_h <= 0:
        return
    scale = max(box_w / hero.width, box_h / hero.height)
    size = (max(box_w, round(hero.width * scale)), max(box_h, round(hero.height * scale)))
    resized = hero.resize(size, Image.Resampling.LANCZOS)
    crop_x = (size[0] - box_w) // 2
    crop_y = (size[1] - box_h) // 2
    cropped = resized.crop((crop_x, crop_y, crop_x + box_w, crop_y + box_h))
    canvas_img.paste(cropped, (x0, y0), cropped if cropped.mode == "RGBA" else None)


def _translucent_band(canvas_img: Image.Image, box: tuple[int, int, int, int], color: _RGB) -> None:
    x0, y0, x1, y1 = box
    overlay = Image.new("RGBA", (x1 - x0, y1 - y0), (*color, 180))
    canvas_img.paste(overlay, (x0, y0), overlay)


# --- composition -------------------------------------------------------------


def _feature_bullets(config: NicheConfig) -> list[str]:
    return [
        f"•  {config.book.page_count} unique hand-crafted designs",
        "•  Single-sided pages prevent bleed-through",
        f"•  {config.style.line_weight.capitalize()} lines, easy to color",
        f"•  Large {config.book.trim_size.replace('x', ' x ')} inch pages",
    ]


def _draw_front(
    canvas_img: Image.Image,
    draw: ImageDraw.ImageDraw,
    layout: CoverLayout,
    *,
    variant: str,
    hero: Image.Image,
    fonts: CoverFonts,
    title: str,
    subtitle: str,
    author: str,
    text_color: _RGB,
    accent: _RGB,
    background: _RGB,
) -> None:
    x0 = layout.front[0] + layout.safe_px
    x1 = layout.front[1] - layout.safe_px
    y0 = layout.bleed_px + layout.safe_px
    y1 = layout.height - layout.bleed_px - layout.safe_px
    width, height = x1 - x0, y1 - y0
    center_x = (x0 + x1) // 2
    headline = title.upper()

    if variant == "c":
        _paste_hero_fill(
            canvas_img,
            hero,
            (layout.front[0], layout.bleed_px, layout.front[1], layout.height - layout.bleed_px),
        )
        band_top = y0 + round(height * 0.30)
        band_height = round(height * 0.36)
        _translucent_band(
            canvas_img,
            (layout.front[0], band_top, layout.front[1], band_top + band_height),
            background,
        )
        title_font = _fit_font(
            draw,
            fonts.title,
            headline,
            max_width=width,
            max_lines=2,
            max_size=round(width * 0.17),
        )
        below = _draw_wrapped(
            draw,
            headline,
            title_font,
            center_x=center_x,
            top_y=band_top + round(band_height * 0.10),
            max_width=width,
            fill=text_color,
        )
        _draw_wrapped(
            draw,
            subtitle,
            _font(fonts.body_bold, round(width * 0.045)),
            center_x=center_x,
            top_y=below + round(height * 0.01),
            max_width=width,
            fill=accent,
        )
    elif variant == "b":
        title_font = _fit_font(
            draw,
            fonts.title,
            headline,
            max_width=width,
            max_lines=4,
            max_size=round(width * 0.21),
        )
        below = _draw_wrapped(
            draw,
            headline,
            title_font,
            center_x=center_x,
            top_y=y0 + round(height * 0.05),
            max_width=width,
            fill=text_color,
        )
        _draw_wrapped(
            draw,
            subtitle,
            _font(fonts.body_bold, round(width * 0.05)),
            center_x=center_x,
            top_y=below + round(height * 0.02),
            max_width=width,
            fill=accent,
        )
        _paste_hero_fit(
            canvas_img,
            hero,
            (x0 + round(width * 0.16), y0 + round(height * 0.44), x1, y1 - round(height * 0.06)),
        )
    else:  # variant a — title on top, hero centred below
        title_font = _fit_font(
            draw,
            fonts.title,
            headline,
            max_width=width,
            max_lines=3,
            max_size=round(width * 0.17),
        )
        below = _draw_wrapped(
            draw,
            headline,
            title_font,
            center_x=center_x,
            top_y=y0 + round(height * 0.03),
            max_width=width,
            fill=text_color,
        )
        below = _draw_wrapped(
            draw,
            subtitle,
            _font(fonts.body_bold, round(width * 0.05)),
            center_x=center_x,
            top_y=below + round(height * 0.015),
            max_width=width,
            fill=accent,
        )
        _paste_hero_fit(
            canvas_img,
            hero,
            (x0, round(below + height * 0.03), x1, y1 - round(height * 0.08)),
        )

    _draw_centered(
        draw,
        author,
        _font(fonts.body_bold, round(width * 0.046)),
        center_x=center_x,
        y=y1 - round(height * 0.055),
        fill=text_color,
    )


def _draw_back(
    canvas_img: Image.Image,
    draw: ImageDraw.ImageDraw,
    layout: CoverLayout,
    config: NicheConfig,
    fonts: CoverFonts,
    title: str,
    author: str,
    text_color: _RGB,
    accent: _RGB,
) -> None:
    x0 = layout.back[0] + layout.safe_px
    x1 = layout.back[1] - layout.safe_px
    y0 = layout.bleed_px + layout.safe_px
    y1 = layout.height - layout.bleed_px - layout.safe_px
    width = x1 - x0
    center_x = (x0 + x1) // 2

    title_font = _fit_font(
        draw,
        fonts.title,
        title.upper(),
        max_width=width,
        max_lines=3,
        max_size=round(width * 0.11),
    )
    y = _draw_wrapped(
        draw,
        title.upper(),
        title_font,
        center_x=center_x,
        top_y=y0,
        max_width=width,
        fill=text_color,
    )
    y = _draw_wrapped(
        draw,
        config.metadata.subtitle_seed,
        _font(fonts.body, round(width * 0.04)),
        center_x=center_x,
        top_y=y + round(width * 0.05),
        max_width=width,
        fill=text_color,
    )
    bullet_font = _font(fonts.body, round(width * 0.036))
    ascent, descent = bullet_font.getmetrics()
    y += round(width * 0.06)
    for bullet in _feature_bullets(config):
        draw.text((x0, y), bullet, font=bullet_font, fill=accent)
        y += (ascent + descent) * 1.4
    _draw_centered(
        draw,
        author,
        _font(fonts.body_bold, round(width * 0.044)),
        center_x=center_x,
        y=y1 - round(width * 0.06),
        fill=text_color,
    )


def _draw_spine(
    canvas_img: Image.Image,
    layout: CoverLayout,
    title: str,
    fonts: CoverFonts,
    text_color: _RGB,
) -> None:
    # KDP forbids spine text on thin spines — skip it below the threshold.
    if layout.dims.spine_width_in < _MIN_SPINE_TEXT_IN:
        return
    spine_width = layout.spine[1] - layout.spine[0]
    spine_height = layout.height - 2 * layout.bleed_px
    strip = Image.new("RGBA", (spine_height, spine_width), (0, 0, 0, 0))
    strip_draw = ImageDraw.Draw(strip)
    font = _font(fonts.title, round(spine_width * 0.6))
    headline = title.upper()
    text_width = strip_draw.textlength(headline, font=font)
    ascent, descent = font.getmetrics()
    strip_draw.text(
        ((spine_height - text_width) / 2, (spine_width - ascent - descent) / 2),
        headline,
        font=font,
        fill=text_color,
    )
    rotated = strip.rotate(90, expand=True)
    canvas_img.paste(rotated, (layout.spine[0], layout.bleed_px), rotated)


def compose_cover(
    book: Book,
    config: NicheConfig,
    layout: CoverLayout,
    *,
    variant: str,
    hero: Image.Image,
    fonts: CoverFonts,
) -> Image.Image:
    """Composite a full wrap cover for one layout variant."""
    background = _hex_rgb(config.cover.background_color)
    text_color = _hex_rgb(config.cover.text_color)
    accent = _hex_rgb(config.cover.accent_color)
    canvas_img = Image.new("RGB", (layout.width, layout.height), background)
    draw = ImageDraw.Draw(canvas_img)

    title = book.title or config.metadata.title_seed
    subtitle = book.subtitle or config.metadata.subtitle_seed
    author = config.metadata.author

    _draw_back(canvas_img, draw, layout, config, fonts, title, author, text_color, accent)
    _draw_spine(canvas_img, layout, title, fonts, text_color)
    _draw_front(
        canvas_img,
        draw,
        layout,
        variant=variant,
        hero=hero,
        fonts=fonts,
        title=title,
        subtitle=subtitle,
        author=author,
        text_color=text_color,
        accent=accent,
        background=background,
    )
    return canvas_img


def _export_pdf(composite: Image.Image, layout: CoverLayout, pdf_path: Path) -> None:
    width_pt = layout.dims.total_width_in * POINTS_PER_INCH
    height_pt = layout.dims.total_height_in * POINTS_PER_INCH
    pdf = canvas.Canvas(str(pdf_path), pagesize=(width_pt, height_pt))
    pdf.drawImage(ImageReader(composite), 0, 0, width=width_pt, height=height_pt)
    pdf.showPage()
    pdf.save()


def build_cover_variant(
    book: Book,
    config: NicheConfig,
    *,
    variant: str,
    hero_path: Path,
    output_dir: Path,
    interior_page_count: int,
    fonts: CoverFonts,
    paper: str = "white",
    dpi: int = 300,
) -> tuple[Path, Path]:
    """Composite one cover variant; return its (PNG preview, PDF) paths."""
    trim_w, trim_h = parse_trim_size(config.book.trim_size)
    layout = cover_layout(interior_page_count, trim_w, trim_h, paper=paper, dpi=dpi)
    with Image.open(hero_path) as handle:
        hero = handle.convert("RGBA")
    composite = compose_cover(book, config, layout, variant=variant, hero=hero, fonts=fonts)

    cover_dir = output_dir / book.slug / "cover"
    cover_dir.mkdir(parents=True, exist_ok=True)
    png_path = cover_dir / f"cover_variant_{variant}.png"
    composite.save(png_path)

    pdf_dir = output_dir / book.slug / "pdf"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = pdf_dir / f"cover_variant_{variant}.pdf"
    _export_pdf(composite, layout, pdf_path)
    return png_path, pdf_path


def build_all_covers(
    book: Book,
    config: NicheConfig,
    *,
    hero_path: Path,
    output_dir: Path,
    interior_page_count: int,
    paper: str = "white",
    dpi: int = 300,
) -> list[Path]:
    """Build all three cover variants; return the list of variant PDF paths."""
    fonts = load_cover_fonts()
    pdfs: list[Path] = []
    for variant in VARIANTS:
        _, pdf_path = build_cover_variant(
            book,
            config,
            variant=variant,
            hero_path=hero_path,
            output_dir=output_dir,
            interior_page_count=interior_page_count,
            fonts=fonts,
            paper=paper,
            dpi=dpi,
        )
        pdfs.append(pdf_path)
        logger.info("cover variant {} written: {}", variant, pdf_path)
    return pdfs
