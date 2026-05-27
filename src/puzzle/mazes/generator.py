"""Deterministic maze generation via mazelib.

Mazelib uses both ``random`` and ``numpy.random`` internally; the ``Maze``
constructor's ``seed=`` argument seeds both, so the maze is fully
deterministic given (algorithm, dimensions, seed). Generation is **serial**
— mazelib's algorithms touch process-wide RNG state, so concurrent runs
would race; for a 40-maze book the entire batch completes in well under
two seconds, which makes parallelism unnecessary.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from mazelib import Maze
from mazelib.generate.BacktrackingGenerator import BacktrackingGenerator
from mazelib.generate.HuntAndKill import HuntAndKill
from mazelib.generate.Kruskal import Kruskal
from mazelib.generate.Prims import Prims
from mazelib.solve.BacktrackingSolver import BacktrackingSolver

from src.puzzle.mazes.models import Difficulty, GeneratedMaze

if TYPE_CHECKING:
    from src.config.schema import PuzzleSpec

Algorithm = Literal["prim", "kruskal", "backtracker", "hunt_and_kill"]

# Maps the niche-config algorithm names to the mazelib generator classes.
# Mazelib's name for the recursive backtracker is `BacktrackingGenerator`;
# its name for Prim's algorithm is `Prims` (note the 's'). The plain English
# names live in the niche YAML so authors don't have to know mazelib internals.
_ALGORITHMS: dict[str, type] = {
    "prim": Prims,
    "kruskal": Kruskal,
    "backtracker": BacktrackingGenerator,
    "hunt_and_kill": HuntAndKill,
}


def generate_maze(
    *,
    grid_size: tuple[int, int],
    algorithm: str,
    seed: int,
    difficulty: Difficulty,
    index: int,
) -> GeneratedMaze:
    """Generate one solvable maze at the given seed.

    ``grid_size`` is ``(width_cells, height_cells)`` — the underlying grid
    is sized ``(2H + 1, 2W + 1)`` to give every cell its own wall. The
    returned maze always has a perimeter start/end on opposite walls and a
    non-empty solution path between them (mazelib's BacktrackingSolver).
    """
    if algorithm not in _ALGORITHMS:
        raise ValueError(
            f"unknown maze algorithm {algorithm!r}; choose one of {sorted(_ALGORITHMS)}"
        )
    width_cells, height_cells = grid_size
    if width_cells <= 0 or height_cells <= 0:
        raise ValueError(f"grid_size cells must be positive: {grid_size}")

    maze = Maze(seed=seed)
    # Mazelib's generators take (H, W) — height first.
    maze.generator = _ALGORITHMS[algorithm](height_cells, width_cells)
    maze.generate()
    # `generate_entrances` defaults to opposite outer walls — exactly what a
    # printed maze wants (a clear "in" and "out" arrow).
    maze.generate_entrances()
    maze.solver = BacktrackingSolver()
    maze.solve()

    if not maze.solutions:
        # Defense in depth — mazelib should never produce an unsolvable maze
        # with the perimeter entrances pattern.
        raise RuntimeError(
            f"mazelib produced an unsolvable maze (seed={seed}, "
            f"algorithm={algorithm}, grid_size={grid_size})"
        )

    grid_tuple: tuple[tuple[int, ...], ...] = tuple(
        tuple(int(cell) for cell in row) for row in maze.grid
    )
    solution_tuple: tuple[tuple[int, int], ...] = tuple(
        (int(r), int(c)) for r, c in maze.solutions[0]
    )
    return GeneratedMaze(
        index=index,
        difficulty=difficulty,
        grid=grid_tuple,
        solution=solution_tuple,
        start=(int(maze.start[0]), int(maze.start[1])),
        end=(int(maze.end[0]), int(maze.end[1])),
        seed=seed,
    )


def generate_book_mazes(spec: PuzzleSpec, *, book_seed_prefix: int) -> list[GeneratedMaze]:
    """Generate every maze for one book — serial, deterministic.

    The seed for maze ``i`` is ``book_seed_prefix * 10000 + i``, matching
    the seeding convention coloring books use in `src/db/models.py:Book.seed_prefix`.
    Difficulty for maze ``i`` cycles through ``spec.difficulty_curve`` —
    ``curve[i % len(curve)]``. The grid size for each maze is looked up
    from ``spec.grid_sizes`` by difficulty.
    """
    mazes: list[GeneratedMaze] = []
    curve = spec.difficulty_curve
    for i in range(spec.count):
        difficulty = curve[i % len(curve)]
        grid_size = spec.grid_sizes[difficulty]
        seed = spec.fixed_seed if spec.fixed_seed is not None else book_seed_prefix * 10000 + i
        mazes.append(
            generate_maze(
                grid_size=grid_size,
                algorithm=spec.algorithm,
                seed=seed,
                difficulty=difficulty,
                index=i,
            )
        )
    return mazes
