"""Cover assembly — hero illustration, composited wrap cover, print-ready PDF.

The cover is one rasterised image (back | spine | front, with bleed) at print
DPI; text is drawn into the pixels with Pillow, so the exported PDF embeds a
single image and carries no fonts. The front panel is the hero art bled to the
trim edges with a cream title plate over it; the back panel carries the blurb,
benefit bullets, and a row of framed interior-page previews.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

from src.config.imprint import load_imprint
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

# Hero illustration — Fal.ai FLUX dev. The endpoint clamps custom sizes to a
# 1536 max, so 1536² is the largest hero it returns: 2.36 MP, billed as the
# 3 MP tier (Fal rounds image area up) → ~$0.075/image at flux/dev's $0.025/MP.
_HERO_MODEL = "fal-ai/flux/dev"
_HERO_PX = 1536
_HERO_STEPS = 28
_HERO_GUIDANCE = 3.5

# KDP forbids spine text below this spine width.
_MIN_SPINE_TEXT_IN = 0.25

_RGB = tuple[int, int, int]


@dataclass(frozen=True, slots=True)
class _Palette:
    """A cover palette: plate fill, accent (keyline / badge / stars), text ink."""

    plate: _RGB
    accent: _RGB
    ink: _RGB


# Named palettes — selected per imprint via `imprint.visual_style.cover_palette_name`.
# Add new entries here; the hero art carries the rest of the niche colour story.
_PALETTES: dict[str, _Palette] = {
    "cottagecore_earth": _Palette(
        plate=(247, 241, 227),
        accent=(181, 92, 60),
        ink=(60, 42, 30),
    ),
    "warm_friendly": _Palette(
        plate=(255, 248, 231),  # #fff8e7  warm cream
        accent=(122, 74, 44),  # #7a4a2c  chocolate brown — keylines + badges
        ink=(196, 90, 58),  # #c45a3a  terracotta-rust — text on plate
    ),
}
_DEFAULT_PALETTE = "cottagecore_earth"


def _resolve_palette(name: str) -> _Palette:
    """Return the named palette; fall back to the default with a warning."""
    if name in _PALETTES:
        return _PALETTES[name]
    logger.warning(
        "cover palette {!r} not defined; falling back to {!r}",
        name,
        _DEFAULT_PALETTE,
    )
    return _PALETTES[_DEFAULT_PALETTE]


@dataclass(frozen=True, slots=True)
class _BarcodeZone:
    """KDP back-cover EAN-barcode keep-out area, in inches.

    KDP overlays the barcode automatically at print time; our cover artwork
    must reserve this rectangle (`width_in` x `height_in`, inset `inset_in`
    from both axes) so the overlay lands on background, not on a thumbnail
    or text. In the flat back|spine|front cover spread we deliver to KDP,
    the zone sits at the back panel's bottom-right (spine-side bottom)
    corner — i.e. the spine-side bottom of the printed back cover when the
    book is closed and viewed face-up.
    """

    width_in: float
    height_in: float
    inset_in: float


# Reserved keep-out for KDP's auto-overlaid EAN barcode on the back cover.
# 2.0" x 1.2", inset 0.25" from the back panel's bottom-right corner in the
# flat spread (= spine-side bottom of the printed back cover). `_barcode_zone_px`
# resolves it to a pixel rect; `_draw_back` caps the thumbnail row so it does
# not intrude.
BARCODE_CLEAR_ZONE = _BarcodeZone(width_in=2.0, height_in=1.2, inset_in=0.25)


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
    """Largest font (<= max_size) at which `text` both wraps within the line
    budget and keeps every line inside `max_width` — a long single word can
    overflow on width while still counting as one 'line'."""
    size = max_size
    while size > 24:
        font = _font(path, size)
        lines = _wrap(draw, text, font, max_width)
        if len(lines) <= max_lines and all(
            draw.textlength(line, font=font) <= max_width for line in lines
        ):
            return font
        size -= 4
    return _font(path, 24)


def _block_height(font: ImageFont.FreeTypeFont, line_count: int, spacing: float) -> float:
    """Pixel height of `line_count` lines drawn by `_draw_wrapped` at `spacing`."""
    ascent, descent = font.getmetrics()
    return line_count * (ascent + descent) * spacing


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


def _draw_star(draw: ImageDraw.ImageDraw, cx: float, cy: float, r: float, fill: _RGB) -> None:
    """Draw a small four-pointed star (a ✦-style bullet marker)."""
    inner = r * 0.40
    points: list[tuple[float, float]] = []
    for i in range(8):
        radius = r if i % 2 == 0 else inner
        angle = -math.pi / 2 + i * math.pi / 4
        points.append((cx + radius * math.cos(angle), cy + radius * math.sin(angle)))
    draw.polygon(points, fill=fill)


# --- hero placement ----------------------------------------------------------


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


# --- composition -------------------------------------------------------------


def _feature_bullets(config: NicheConfig) -> list[str]:
    """Fallback back-cover bullets when the niche sets none of its own."""
    return [
        f"{config.book.page_count} original designs",
        "Bold, easy-to-color outlines for stress-free coloring",
        "Single-sided pages prevent bleed-through",
        f"Large {config.book.trim_size.replace('x', ' x ')} inch format",
    ]


def _draw_front(
    canvas_img: Image.Image,
    draw: ImageDraw.ImageDraw,
    layout: CoverLayout,
    config: NicheConfig,
    *,
    hero: Image.Image,
    fonts: CoverFonts,
    palette: _Palette,
) -> None:
    """Front panel — hero bled to the trim edges, a cream title plate, a badge."""
    fx0, fx1 = layout.front
    panel_w = fx1 - fx0
    height = layout.height
    center_x = (fx0 + fx1) // 2
    safe = layout.safe_px

    # Hero fills the panel, bleeding off the right, top and bottom trim edges.
    _paste_hero_fill(canvas_img, hero, (fx0, 0, layout.width, height))

    headline = config.cover.headline or config.metadata.title_seed.upper()
    subtitle = config.cover.subtitle or config.metadata.subtitle_seed
    badge = config.cover.badge_text

    # --- title plate: a compact cream card flush with the top safe margin ----
    plate_w = round(panel_w * 0.74)
    plate_x0 = center_x - plate_w // 2
    plate_x1 = plate_x0 + plate_w
    pad_x = round(plate_w * 0.050)
    pad_y = round(plate_w * 0.030)
    gap = round(plate_w * 0.012)
    inner_w = plate_w - 2 * pad_x

    # The headline is sized so its longest line spans ~80% of the plate width.
    headline_font = _fit_font(
        draw,
        fonts.title_black,
        headline,
        max_width=round(plate_w * 0.82),
        max_lines=2,
        max_size=round(plate_w * 0.46),
    )
    subtitle_font = _font(fonts.title_regular, round(plate_w * 0.043))

    h_lines = len(_wrap(draw, headline, headline_font, inner_w))
    s_lines = len(_wrap(draw, subtitle, subtitle_font, inner_w))
    h_block = _block_height(headline_font, h_lines, 0.96)
    s_block = _block_height(subtitle_font, s_lines, 1.05)

    plate_h = round(2 * pad_y + h_block + gap + s_block)
    plate_y0 = layout.bleed_px + safe
    plate_y1 = plate_y0 + plate_h

    draw.rounded_rectangle(
        (plate_x0, plate_y0, plate_x1, plate_y1),
        radius=round(plate_w * 0.028),
        fill=palette.plate,
        outline=palette.accent,
        width=max(3, round(panel_w * 0.003)),
    )

    below = _draw_wrapped(
        draw,
        headline,
        headline_font,
        center_x=center_x,
        top_y=plate_y0 + pad_y,
        max_width=inner_w,
        fill=palette.ink,
        spacing=0.96,
    )
    _draw_wrapped(
        draw,
        subtitle,
        subtitle_font,
        center_x=center_x,
        top_y=below + gap,
        max_width=inner_w,
        fill=palette.ink,
        spacing=1.05,
    )

    # --- corner badge: a terracotta pill in the bottom-right -----------------
    if badge:
        badge_font = _font(fonts.title, round(panel_w * 0.020))
        b_ascent, b_descent = badge_font.getmetrics()
        b_pad_x = round(panel_w * 0.019)
        b_pad_y = round(panel_w * 0.010)
        badge_w = round(draw.textlength(badge, font=badge_font)) + 2 * b_pad_x
        badge_h = b_ascent + b_descent + 2 * b_pad_y
        badge_x1 = fx1 - safe
        badge_x0 = badge_x1 - badge_w
        badge_y1 = height - layout.bleed_px - safe
        badge_y0 = badge_y1 - badge_h
        draw.rounded_rectangle(
            (badge_x0, badge_y0, badge_x1, badge_y1),
            radius=badge_h // 2,
            fill=palette.accent,
        )
        draw.text(
            (badge_x0 + b_pad_x, badge_y0 + b_pad_y),
            badge,
            font=badge_font,
            fill=palette.plate,
        )


def _draw_thumbnails(
    canvas_img: Image.Image,
    draw: ImageDraw.ImageDraw,
    *,
    images: list[Path],
    x0: int,
    x1: int,
    top_y: int,
    cell: int,
    palette: _Palette,
) -> None:
    """Paste a centred row of framed interior-page previews."""
    if not images or cell <= 0:
        return
    gap = round((x1 - x0) * 0.038)
    total = len(images) * cell + (len(images) - 1) * gap
    start_x = x0 + ((x1 - x0) - total) // 2
    border = max(3, round(cell * 0.022))
    inset = round(cell * 0.055)
    inner = cell - 2 * inset
    for idx, path in enumerate(images):
        cx = start_x + idx * (cell + gap)
        draw.rounded_rectangle(
            (cx, top_y, cx + cell, top_y + cell),
            radius=round(cell * 0.06),
            fill=palette.plate,
            outline=palette.accent,
            width=border,
        )
        with Image.open(path) as handle:
            page = handle.convert("RGB").resize((inner, inner), Image.Resampling.LANCZOS)
        canvas_img.paste(page, (cx + inset, top_y + inset))


def _barcode_zone_px(layout: CoverLayout) -> tuple[int, int, int, int]:
    """Back-cover barcode keep-out rectangle (x0, y0, x1, y1) in canvas pixels."""
    inset = round(BARCODE_CLEAR_ZONE.inset_in * layout.dpi)
    zone_w = round(BARCODE_CLEAR_ZONE.width_in * layout.dpi)
    zone_h = round(BARCODE_CLEAR_ZONE.height_in * layout.dpi)
    x1 = layout.back[1] - inset
    y1 = layout.height - layout.bleed_px - inset
    return x1 - zone_w, y1 - zone_h, x1, y1


def _draw_back(
    canvas_img: Image.Image,
    draw: ImageDraw.ImageDraw,
    layout: CoverLayout,
    config: NicheConfig,
    fonts: CoverFonts,
    title: str,
    author: str,
    text_color: _RGB,
    thumbnails: list[Path],
    palette: _Palette,
) -> None:
    x0 = layout.back[0] + layout.safe_px
    x1 = layout.back[1] - layout.safe_px
    y0 = layout.bleed_px + layout.safe_px
    y1 = layout.height - layout.bleed_px - layout.safe_px
    width = x1 - x0
    center_x = (x0 + x1) // 2

    # Headline: niche override (mixed case, sells the experience) or the title
    # in all caps as a fallback. One line keeps the lower band free for previews.
    headline = config.cover.back_headline or title.upper()
    max_lines = 2 if config.cover.back_headline else 1
    title_font = _fit_font(
        draw,
        fonts.title,
        headline,
        max_width=width,
        max_lines=max_lines,
        max_size=round(width * 0.13),
    )
    y = _draw_wrapped(
        draw,
        headline,
        title_font,
        center_x=center_x,
        top_y=y0,
        max_width=width,
        fill=text_color,
        spacing=1.0,
    )

    if config.cover.tagline:
        y += round(width * 0.035)
        y = _draw_wrapped(
            draw,
            config.cover.tagline,
            _font(fonts.body_bold, round(width * 0.037)),
            center_x=center_x,
            top_y=y,
            max_width=width,
            fill=text_color,
            spacing=1.22,
        )

    bullets = config.cover.bullets or _feature_bullets(config)
    bullet_font = _font(fonts.body_bold, round(width * 0.034))
    b_ascent, b_descent = bullet_font.getmetrics()
    b_line = (b_ascent + b_descent) * 1.42
    star_r = round(width * 0.016)
    y += round(width * 0.05)
    for bullet in bullets:
        mid = y + (b_ascent + b_descent) / 2
        _draw_star(draw, x0 + star_r, mid, star_r, palette.accent)
        draw.text((x0 + round(width * 0.06), y), bullet, font=bullet_font, fill=text_color)
        y += b_line

    author_font = _font(fonts.body_bold, round(width * 0.044))
    author_y = y1 - round(width * 0.05)

    # Interior-page previews fill the band between the bullets and the author,
    # held clear of the KDP barcode zone in the back cover's bottom-right.
    if thumbnails:
        _, barcode_top, _, _ = _barcode_zone_px(layout)
        band_top = y + round(width * 0.035)
        band_bot = min(author_y - round(width * 0.035), barcode_top - round(width * 0.025))
        gap = round(width * 0.038)
        cell = int(min((width - 2 * gap) // 3, band_bot - band_top, round(width * 0.30)))
        if cell > 80:
            row_top = int(band_top + ((band_bot - band_top) - cell) // 2)
            _draw_thumbnails(
                canvas_img,
                draw,
                images=thumbnails,
                x0=x0,
                x1=x1,
                top_y=row_top,
                cell=cell,
                palette=palette,
            )

    _draw_centered(draw, author, author_font, center_x=center_x, y=author_y, fill=text_color)


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
    hero: Image.Image,
    fonts: CoverFonts,
    thumbnails: list[Path],
    palette: _Palette,
) -> Image.Image:
    """Composite the full wrap cover — back, spine, and front."""
    background = _hex_rgb(config.cover.background_color)
    text_color = _hex_rgb(config.cover.text_color)
    canvas_img = Image.new("RGB", (layout.width, layout.height), background)
    draw = ImageDraw.Draw(canvas_img)

    title = config.cover.headline or book.title or config.metadata.title_seed
    author = config.metadata.author

    _draw_back(
        canvas_img, draw, layout, config, fonts, title, author, text_color, thumbnails, palette
    )
    _draw_spine(canvas_img, layout, title, fonts, text_color)
    _draw_front(canvas_img, draw, layout, config, hero=hero, fonts=fonts, palette=palette)
    return canvas_img


def _export_pdf(composite: Image.Image, layout: CoverLayout, pdf_path: Path) -> None:
    width_pt = layout.dims.total_width_in * POINTS_PER_INCH
    height_pt = layout.dims.total_height_in * POINTS_PER_INCH
    pdf = canvas.Canvas(str(pdf_path), pagesize=(width_pt, height_pt))
    pdf.drawImage(ImageReader(composite), 0, 0, width=width_pt, height=height_pt)
    pdf.showPage()
    pdf.save()


def build_cover(
    book: Book,
    config: NicheConfig,
    *,
    hero_path: Path,
    output_dir: Path,
    interior_page_count: int,
    paper: str = "white",
    dpi: int = 300,
) -> Path:
    """Composite the wrap cover; return its print-ready PDF path."""
    imprint = load_imprint(config.imprint)
    fonts = load_cover_fonts(
        primary_family=imprint.visual_style.cover_font_primary,
        secondary_family=imprint.visual_style.cover_font_secondary,
    )
    palette = _resolve_palette(imprint.visual_style.cover_palette_name)
    trim_w, trim_h = parse_trim_size(config.book.trim_size)
    layout = cover_layout(interior_page_count, trim_w, trim_h, paper=paper, dpi=dpi)
    with Image.open(hero_path) as handle:
        hero = handle.convert("RGBA")

    filtered = sorted((output_dir / book.slug / "images" / "filtered").glob("*.png"))
    thumbnails = [filtered[i] for i in config.cover.thumbnails if 0 <= i < len(filtered)]
    composite = compose_cover(
        book, config, layout, hero=hero, fonts=fonts, thumbnails=thumbnails, palette=palette
    )

    cover_dir = output_dir / book.slug / "cover"
    cover_dir.mkdir(parents=True, exist_ok=True)
    composite.save(cover_dir / "cover.png")

    pdf_dir = output_dir / book.slug / "pdf"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = pdf_dir / "cover.pdf"
    _export_pdf(composite, layout, pdf_path)
    logger.info("cover written: {}", pdf_path)
    return pdf_path
