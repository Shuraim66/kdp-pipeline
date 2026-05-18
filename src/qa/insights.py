"""Feedback-loop analytics — turn QA history into actionable insight.

Pure aggregation and rendering: every function here takes already-fetched
rows or domain objects and returns a formatted report string. No database
access, no IO — the SQL lives in `src/db/repos/images.py`, the CLI wiring in
`src/main.py`. This mirrors `build_vision_qa_summary` in `vision_qa.py` and
keeps the reports unit-testable without a database.

The four reports back the Q6 feedback-loop CLI commands:

  * `build_prompt_hint_report`         -> ``analyze-prompt-hints``
  * `build_subject_performance_report` -> ``subject-performance``
  * `build_ink_density_report`         -> ``ink-density``
  * `build_probe_report`               -> ``probe-vision-qa``
"""

from __future__ import annotations

import re
import statistics
from collections import Counter, defaultdict
from decimal import Decimal
from itertools import pairwise
from typing import Any

from src.db.models import Image
from src.qa.vision_qa import VisionQAResult

# Rubric vocabulary — the words Claude's prompt hints inherit from the Vision
# QA rubric itself ("improve the LINE QUALITY", "the SUBJECT should ..."). They
# describe the grading criteria, not the page, so counting them drowns out the
# content-specific feedback. Excluded from the keyword frequency pass.
_RUBRIC_STOPWORDS = frozenset(
    {
        "line",
        "lines",
        "subject",
        "subjects",
        "composition",
        "compositions",
        "anatomy",
        "whitespace",
        "score",
        "scores",
        "page",
        "pages",
        "image",
        "images",
        "illustration",
        "illustrations",
        "character",
        "characters",
        "object",
        "objects",
        "scene",
        "scenes",
        "niche",
    }
)
# Ordinary English filler plus the generic imperative verbs that open a hint
# sentence ("Add a ...", "Make the ...", "Consider ...") — no content either.
_GENERAL_STOPWORDS = frozenset(
    {
        "the",
        "and",
        "for",
        "with",
        "that",
        "this",
        "its",
        "are",
        "was",
        "were",
        "have",
        "has",
        "had",
        "but",
        "not",
        "too",
        "very",
        "more",
        "less",
        "most",
        "should",
        "would",
        "could",
        "make",
        "makes",
        "made",
        "making",
        "add",
        "adds",
        "added",
        "adding",
        "use",
        "uses",
        "using",
        "into",
        "onto",
        "from",
        "out",
        "off",
        "all",
        "any",
        "some",
        "than",
        "then",
        "them",
        "they",
        "you",
        "your",
        "can",
        "may",
        "will",
        "one",
        "two",
        "few",
        "look",
        "looks",
        "like",
        "such",
        "around",
        "between",
        "consider",
        "ensure",
        "try",
        "keep",
        "give",
        "place",
        "show",
        "include",
        "there",
        "their",
        "which",
        "while",
        "when",
        "where",
        "what",
        "also",
        "still",
        "just",
        "each",
        "both",
    }
)
_STOPWORDS = _RUBRIC_STOPWORDS | _GENERAL_STOPWORDS

# Tokens are alphabetic runs only — digits and punctuation are noise here.
_WORD_RE = re.compile(r"[a-z]+")
_MIN_TOKEN_LEN = 3

# Subscore display order for the variance probe — matches the Vision QA rubric.
_SUBSCORE_ORDER = (
    "subject_recognition",
    "composition_coherence",
    "line_quality",
    "anatomy_accuracy",
    "whitespace_balance",
)


# --- shared helpers -------------------------------------------------------


def _bar(count: int, max_count: int, width: int = 24) -> str:
    """A proportional block-bar for a histogram row."""
    if max_count <= 0:
        return ""
    return "█" * max(1, round(width * count / max_count))


def _subject_name(image: Image) -> str:
    """Recover a slot's subject from its stored generation params."""
    subject = image.generation_params.get("subject")
    return str(subject) if subject else image.prompt[:48]


def _truncate(text: str, width: int) -> str:
    """Clip `text` to `width`, ending in an ellipsis when it overruns."""
    return text if len(text) <= width else text[: width - 1] + "…"


# --- analyze-prompt-hints -------------------------------------------------


def _content_tokens(text: str) -> list[str]:
    """Lowercase content words of a hint — stopwords and short tokens dropped."""
    return [
        token
        for token in _WORD_RE.findall(text.lower())
        if len(token) >= _MIN_TOKEN_LEN and token not in _STOPWORDS
    ]


def rank_hint_terms(hints: list[str], *, top_n: int) -> list[tuple[str, int]]:
    """Rank recurring content terms (unigrams + bigrams) across prompt hints.

    Rubric vocabulary is filtered out first (see `_RUBRIC_STOPWORDS`), so what
    surfaces is content-specific feedback. Only terms that recur (count >= 2)
    are returned — a term seen once is noise for a *recurring*-theme report.
    """
    counter: Counter[str] = Counter()
    for hint in hints:
        tokens = _content_tokens(hint)
        counter.update(tokens)
        counter.update(f"{first} {second}" for first, second in pairwise(tokens))
    return [(term, count) for term, count in counter.most_common() if count >= 2][:top_n]


def _normalise_hint(hint: str) -> str:
    """Collapse whitespace, lowercase, and drop a trailing period — for grouping."""
    return " ".join(hint.lower().split()).rstrip(".")


def build_prompt_hint_report(slug: str, images: list[Image], *, top_n: int = 10) -> str:
    """Render the ``analyze-prompt-hints`` report for a book.

    Two views of the Vision QA prompt hints: a keyword-frequency pass that
    surfaces recurring *themes* even when no two hints are worded alike, and a
    verbatim-repeat pass that catches the same canned hint across pages.
    """
    evaluated = [img for img in images if img.vision_qa_score is not None]
    hinted = [img for img in images if img.vision_qa_prompt_hint]
    lines = [f"Prompt-hint analysis — {slug}", "=" * 48]
    if not evaluated:
        lines.append("No images have been vision-evaluated yet.")
        return "\n".join(lines)
    lines.append(f"{len(evaluated)} image(s) evaluated · {len(hinted)} carry a prompt hint")
    if not hinted:
        lines.append("")
        lines.append("No prompt hints recorded — nothing to aggregate.")
        return "\n".join(lines)

    hints = [str(img.vision_qa_prompt_hint) for img in hinted]
    terms = rank_hint_terms(hints, top_n=top_n)
    lines.append("")
    lines.append("Recurring themes (keyword frequency, rubric words filtered):")
    if terms:
        widest = max(count for _, count in terms)
        for term, count in terms:
            lines.append(f"  {term:<26}{count:>3}  {_bar(count, widest)}")
    else:
        lines.append("  (no term recurred — every hint was phrased uniquely)")

    groups: dict[str, list[Image]] = defaultdict(list)
    for img in hinted:
        groups[_normalise_hint(str(img.vision_qa_prompt_hint))].append(img)
    repeats = sorted(
        ((hint, imgs) for hint, imgs in groups.items() if len(imgs) > 1),
        key=lambda pair: len(pair[1]),
        reverse=True,
    )
    lines.append("")
    lines.append("Repeated verbatim (normalised):")
    if repeats:
        for hint, imgs in repeats:
            pages = ",".join(f"{p:03d}" for p in sorted({img.sequence_num for img in imgs}))
            scored = [int(img.vision_qa_score) for img in imgs if img.vision_qa_score is not None]
            avg = f"avg {statistics.mean(scored):.0f}" if scored else "avg —"
            lines.append(f'  {len(imgs)}x  "{hint}"')
            lines.append(f"        pages {pages}  ·  {avg}")
    else:
        lines.append("  (every hint was unique — see the keyword themes above)")

    singles = sum(1 for imgs in groups.values() if len(imgs) == 1)
    if singles:
        lines.append("")
        lines.append(f"({singles} further hint(s) occurred once each.)")
    return "\n".join(lines)


# --- subject-performance --------------------------------------------------


def _top_rejection(tallies: list[tuple[str, int]]) -> str:
    """The most common rejection status of a subject, as ``vision_lines (3)``."""
    if not tallies:
        return "—"
    status, count = max(tallies, key=lambda pair: pair[1])
    return f"{status.removeprefix('rejected_')} ({count})"


def build_subject_performance_report(
    niche: str,
    perf_rows: list[dict[str, Any]],
    rejection_rows: list[dict[str, Any]],
) -> str:
    """Render the ``subject-performance`` report — per-subject, cross-book.

    `perf_rows` and `rejection_rows` are the rows from
    `subject_performance_rows` / `subject_rejection_rows`; this function only
    formats them. Subjects arrive worst-scoring first.
    """
    rejections: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for row in rejection_rows:
        rejections[str(row["subject"])].append((str(row["qa_status"]), int(row["n"])))

    total_slots = sum(int(row["slots"]) for row in perf_rows)
    total_books = max((int(row["books"]) for row in perf_rows), default=0)
    lines = [
        f"Subject performance — niche: {niche}",
        "=" * 78,
        f"{len(perf_rows)} subject(s) · {total_books} book(s) · "
        f"{total_slots} slot(s) · worst-scoring first",
        "",
        f"{'subject':<44}{'slots':>6}{'avg':>6}{'pass%':>7}{'retry%':>8}{'ink%':>7}  top rejection",
    ]
    for row in perf_rows:
        subject = str(row["subject"])
        avg = row["avg_score"]
        ink = row["avg_ink_pct"]
        avg_str = f"{float(avg):.1f}" if avg is not None else "—"
        ink_str = f"{float(ink):.1f}" if ink is not None else "—"
        lines.append(
            f"{_truncate(subject, 43):<44}"
            f"{int(row['slots']):>6}"
            f"{avg_str:>6}"
            f"{int(row['pass_pct']):>6}%"
            f"{int(row['retry_pct']):>7}%"
            f"{ink_str:>7}  "
            f"{_top_rejection(rejections.get(subject, []))}"
        )
    return "\n".join(lines)


# --- ink-density ----------------------------------------------------------


def _ink_row(pct: float, image: Image) -> str:
    """One page line for the ink-density report."""
    return f"  seq {image.sequence_num:03d}  {pct:6.2f}%  {_subject_name(image)}"


def build_ink_density_report(slug: str, images: list[Image], band: tuple[float, float]) -> str:
    """Render the ``ink-density`` report — pages outside the advisory band.

    One row per slot: the latest attempt is the page that counts. The band is
    advisory only — this report flags pages for human review, nothing here
    gates QA.
    """
    low, high = band
    lines = [
        f"Ink density — {slug}",
        "=" * 48,
        f"advisory band: {low:.1f}-{high:.1f}%  (advisory only — does not gate QA)",
    ]
    latest: dict[int, Image] = {}
    for img in images:
        current = latest.get(img.sequence_num)
        if current is None or img.retry_attempt > current.retry_attempt:
            latest[img.sequence_num] = img
    measured = [
        (img.ink_density_pct, img) for img in latest.values() if img.ink_density_pct is not None
    ]
    if not measured:
        lines.append("")
        lines.append("No pages carry an ink-density measurement yet.")
        lines.append("(ink_density_pct is recorded only for pages generated after Q6.)")
        return "\n".join(lines)

    dense = sorted((p for p in measured if p[0] > high), key=lambda p: p[0], reverse=True)
    sparse = sorted((p for p in measured if p[0] < low), key=lambda p: p[0])
    values = [pct for pct, _ in measured]
    lines.append(
        f"{len(measured)} page(s) · {len(dense)} dense · {len(sparse)} sparse · "
        f"{len(measured) - len(dense) - len(sparse)} in band"
    )
    if dense:
        lines.append("")
        lines.append(f"  DENSE  (> {high:.1f}% — busy, risks over-inking)")
        lines += [_ink_row(pct, img) for pct, img in dense]
    if sparse:
        lines.append("")
        lines.append(f"  SPARSE  (< {low:.1f}% — thin or small, weak on the page)")
        lines += [_ink_row(pct, img) for pct, img in sparse]
    lines.append("")
    lines.append(
        f"  median {statistics.median(values):.2f}%  ·  "
        f"min {min(values):.2f}%  ·  max {max(values):.2f}%"
    )
    return "\n".join(lines)


# --- probe-vision-qa ------------------------------------------------------


def _probe_verdict(result: VisionQAResult) -> str:
    """A short pass / reject label for one probe run."""
    if result.passed:
        return "PASS"
    reason = str(result.rejection_reason or "")
    reason = reason.removeprefix("rejected_vision_").removeprefix("rejected_")
    return f"reject ({reason})" if reason else "reject"


def build_probe_report(
    image_name: str,
    subject: str,
    results: list[VisionQAResult],
    *,
    threshold: int,
) -> str:
    """Render the ``probe-vision-qa`` report — Vision QA score variance.

    `results` is the verdict of grading one image `len(results)` times. A run
    set that straddles the pass threshold is flagged: the score is unstable
    there, and a single grading is not trustworthy.
    """
    lines = [
        f"Vision QA variance probe — {image_name}",
        "=" * 48,
        f'subject: "{subject}"  ·  {len(results)} run(s)',
        "",
        f"{'run':>4}  {'score':>5}  {'subj/comp/line/anat/white':<26}verdict",
    ]
    for run, result in enumerate(results, start=1):
        subs = "/".join(str(result.subscores.get(key, 0)) for key in _SUBSCORE_ORDER)
        lines.append(f"{run:>4}  {result.score:>5}  {subs:<26}{_probe_verdict(result)}")

    scores = [result.score for result in results]
    stdev = statistics.stdev(scores) if len(scores) > 1 else 0.0
    low, high = min(scores), max(scores)
    lines.append("")
    lines.append(
        f"score: mean {statistics.mean(scores):.1f}  stdev {stdev:.1f}  "
        f"min {low}  max {high}  range {high - low}"
    )
    passes = sum(1 for result in results if result.passed)
    fails = len(results) - passes
    verdict = f"verdict: {passes} PASS / {fails} reject"
    if passes and fails:
        verdict += f"  ⚠ straddles the threshold ({threshold}) — score is not stable here"
    lines.append(verdict)
    total = sum((result.cost_usd for result in results), Decimal(0))
    lines.append(f"cost: ${total:.4f}")
    return "\n".join(lines)
