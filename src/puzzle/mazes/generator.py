"""Deterministic maze generation via mazelib.

Mazelib uses both ``random`` and ``numpy.random`` internally; the ``Maze``
constructor's ``seed=`` argument seeds both, so the maze is fully
deterministic given (algorithm, dimensions, seed). Generation is **serial**
— mazelib's algorithms touch process-wide RNG state, so concurrent runs
would race; for a 40-maze book the entire batch completes in well under
two seconds, which makes parallelism unnecessary.

Mazelib's own ``_generate_outer_entrances`` picks one wall axis (top-bottom
or left-right) and then chooses each entrance's COLUMN/ROW uniformly at
random along the perimeter. That's enough to put start/end on *opposite*
walls but not enough to guarantee they're *diagonally* far apart, which is
what makes a maze actually wind. We override mazelib's pick with a
guaranteed-diagonal placement and then validate the solution length —
short-solution mazes regenerate with the next seed.
"""

from __future__ import annotations

import random
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

# Minimum solution length, in cells, relative to the larger grid dimension.
# A solution shorter than this on a `max_cells x max_cells` grid means the
# path skimmed a corridor instead of winding — the maze is too trivial.
_MIN_SOLUTION_PER_GRID_CELL = 1.5

# Hard floor — even toddler 5x5 mazes need at least this many cells in the
# solution path to feel like a real maze rather than two connected rooms.
_MIN_SOLUTION_FLOOR = 6

# How many seed bumps to try when the natural seed produces a too-short
# solution. Empirically the first attempt nearly always passes (diagonal
# entrances make winding paths the rule, not the exception); 20 attempts
# is far more than realistic configs ever need.
_MAX_REGENERATE_ATTEMPTS = 20


def _pick_diagonal_entrances(
    grid_shape: tuple[int, int], rng: random.Random
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Place start/end on opposite walls AND opposite ends along that wall.

    Picks one of four diagonal patterns (TL→BR, TR→BL, LT→RB, LB→RT) to
    keep visual variety across a 40-maze book while guaranteeing that
    start and end are far apart on BOTH axes — a solution can't skim a
    straight corridor when it has to traverse most of the grid in both
    dimensions. Slot choices are restricted to the outer third of each
    wall so the diagonal is always meaningful, not just two cells apart.
    """
    h, w = grid_shape
    cols = list(range(1, w, 2))  # passage columns on the top/bottom walls
    rows = list(range(1, h, 2))  # passage rows on the left/right walls
    # Keep at least one slot in each third even on tiny (5x5) grids.
    col_third = max(1, len(cols) // 3)
    row_third = max(1, len(rows) // 3)
    near_left_cols = cols[:col_third]
    near_right_cols = cols[-col_third:]
    near_top_rows = rows[:row_third]
    near_bot_rows = rows[-row_third:]

    pattern = rng.randrange(4)
    if pattern == 0:
        # Top-left entrance → Bottom-right exit
        start = (0, rng.choice(near_left_cols))
        end = (h - 1, rng.choice(near_right_cols))
    elif pattern == 1:
        # Top-right entrance → Bottom-left exit
        start = (0, rng.choice(near_right_cols))
        end = (h - 1, rng.choice(near_left_cols))
    elif pattern == 2:
        # Left-top entrance → Right-bottom exit
        start = (rng.choice(near_top_rows), 0)
        end = (rng.choice(near_bot_rows), w - 1)
    else:
        # Left-bottom entrance → Right-top exit
        start = (rng.choice(near_bot_rows), 0)
        end = (rng.choice(near_top_rows), w - 1)
    return start, end


def _min_solution_length(grid_size: tuple[int, int]) -> int:
    """Reject mazes whose solution is shorter than this — they're too trivial."""
    return max(_MIN_SOLUTION_FLOOR, int(max(grid_size) * _MIN_SOLUTION_PER_GRID_CELL))


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
    returned maze always has a perimeter start/end placed DIAGONALLY (so
    the solution can't skim a corridor) and a winding solution path of at
    least ``_min_solution_length(grid_size)`` cells. If the natural seed
    produces a too-short path the generator retries with seed+1, seed+2,
    ... up to ``_MAX_REGENERATE_ATTEMPTS`` times; the returned
    ``GeneratedMaze.seed`` records whichever attempt actually succeeded.
    """
    if algorithm not in _ALGORITHMS:
        raise ValueError(
            f"unknown maze algorithm {algorithm!r}; choose one of {sorted(_ALGORITHMS)}"
        )
    width_cells, height_cells = grid_size
    if width_cells <= 0 or height_cells <= 0:
        raise ValueError(f"grid_size cells must be positive: {grid_size}")

    min_solution_cells = _min_solution_length(grid_size)
    last_maze = None
    last_actual_seed = seed % (2**32)

    for attempt in range(_MAX_REGENERATE_ATTEMPTS):
        actual_seed = (seed + attempt) % (2**32)
        # Mazelib seeds numpy.random which requires a uint32; book_seed_prefix
        # is 24-bit and i is small, but the product can overflow uint32 for
        # higher prefixes. Modulo keeps the seed in range while preserving
        # determinism for any (prefix, i) pair within one book.
        maze = Maze(seed=actual_seed)
        # Mazelib's generators take (H, W) — height first.
        maze.generator = _ALGORITHMS[algorithm](height_cells, width_cells)
        maze.generate()

        # Override mazelib's random entrance placement with guaranteed-diagonal
        # picks. `random.Random(actual_seed)` is independent of mazelib's
        # internal RNG so this doesn't perturb the maze structure.
        entrance_rng = random.Random(actual_seed)
        start, end = _pick_diagonal_entrances(maze.grid.shape, entrance_rng)
        maze.start = start
        maze.end = end
        # Carve the entrance cells in the grid so the solver can enter / exit.
        # The renderer's _draw_grid skip-paints these cells anyway, so this
        # change only affects the solver's traversal and doesn't double-open
        # any wall.
        maze.grid[start] = 0
        maze.grid[end] = 0

        maze.solver = BacktrackingSolver()
        maze.solve()

        last_maze = maze
        last_actual_seed = actual_seed
        if maze.solutions and len(maze.solutions[0]) >= min_solution_cells:
            break

    if last_maze is None or not last_maze.solutions:
        raise RuntimeError(
            f"mazelib produced no solvable maze for seed={seed}, "
            f"algorithm={algorithm}, grid_size={grid_size} after "
            f"{_MAX_REGENERATE_ATTEMPTS} attempts"
        )

    grid_tuple: tuple[tuple[int, ...], ...] = tuple(
        tuple(int(cell) for cell in row) for row in last_maze.grid
    )
    solution_tuple: tuple[tuple[int, int], ...] = tuple(
        (int(r), int(c)) for r, c in last_maze.solutions[0]
    )
    return GeneratedMaze(
        index=index,
        difficulty=difficulty,
        grid=grid_tuple,
        solution=solution_tuple,
        start=(int(last_maze.start[0]), int(last_maze.start[1])),
        end=(int(last_maze.end[0]), int(last_maze.end[1])),
        seed=last_actual_seed,
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
