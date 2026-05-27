"""Immutable data types passed between the generator and the renderer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Difficulty = Literal["easy", "medium", "hard"]


@dataclass(frozen=True, slots=True)
class GeneratedMaze:
    """One generated maze plus its solution — what the renderer consumes.

    ``grid`` is a 2D tuple of ints (1 = wall, 0 = passage), shape
    ``(2H + 1, 2W + 1)`` for an H x W maze in mazelib's convention. ``start``
    and ``end`` are perimeter cells; ``solution`` is the list of cells from
    just-inside-start to just-inside-end (as produced by mazelib's
    BacktrackingSolver — does NOT include start/end themselves).
    """

    index: int
    difficulty: Difficulty
    grid: tuple[tuple[int, ...], ...]
    solution: tuple[tuple[int, int], ...]
    start: tuple[int, int]
    end: tuple[int, int]
    seed: int

    @property
    def height(self) -> int:
        """Grid height in cells (2H + 1 for an H x W maze)."""
        return len(self.grid)

    @property
    def width(self) -> int:
        """Grid width in cells (2W + 1 for an H x W maze)."""
        return len(self.grid[0]) if self.grid else 0
