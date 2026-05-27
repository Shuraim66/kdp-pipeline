"""Trivial QA for algorithmic puzzle pages — dimensions + presence check.

Puzzle pages don't go through pixel/composition/vision QA (no images table
row, no Fal generation). This module is the puzzle-pipeline equivalent of
src/qa/runner.py:_qa_pending — every page must exist on disk and match the
expected resolution. If anything fails, the book moves to FAILED with the
phase set to "puzzle_qa".
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image


def validate_puzzles(
    *,
    maze_paths: list[Path],
    solution_paths: list[Path],
    expected_dimensions: tuple[int, int],
) -> list[str]:
    """Return a list of issues; an empty list means all puzzles passed.

    Checks every PNG opens, matches ``expected_dimensions`` exactly, and is
    the correct count. The check is intentionally minimal — algorithmic
    output should never drift from these invariants. A failure here is a
    bug in the generator/renderer, not a content problem.
    """
    issues: list[str] = []
    expected_w, expected_h = expected_dimensions

    if len(maze_paths) != len(solution_paths) and solution_paths:
        issues.append(
            f"maze/solution count mismatch: {len(maze_paths)} mazes vs "
            f"{len(solution_paths)} solutions"
        )

    def _check(paths: list[Path], label: str) -> None:
        for path in paths:
            if not path.is_file():
                issues.append(f"{label} missing on disk: {path}")
                continue
            try:
                with Image.open(path) as img:
                    width, height = img.size
            except OSError as exc:
                issues.append(f"{label} not a readable image: {path} ({exc})")
                continue
            if (width, height) != (expected_w, expected_h):
                issues.append(
                    f"{label} wrong dimensions {width}x{height} "
                    f"(expected {expected_w}x{expected_h}): {path}"
                )

    _check(maze_paths, "maze")
    _check(solution_paths, "solution")
    return issues
