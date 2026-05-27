"""Tests for the puzzle build orchestrator — focused on the Gate 2 pause."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
from src.db.models import BookStatus
from src.puzzle import orchestrator

_PUZZLE_YAML_DICT: dict[str, Any] = {
    "slug": "orch_test_v1",
    "niche": "test_mazes",
    "book_type": "puzzle_maze",
    "book": {
        "trim_size": "8.5x11",
        "page_count": 24,
        "price_usd": 6.99,
        "target_audience": "Kids",
    },
    "metadata": {
        "title_seed": "T",
        "subtitle_seed": "S",
        "author": "Studio",
        "keywords_seed": ["kw"],
        "categories": ["Books > A > B", "Books > C > D"],
    },
    "cover": {
        "background_color": "#fff8e7",
        "accent_color": "#c45a3a",
        "text_color": "#2b1810",
        "hero_subject": "h",
        "hero_style": "s",
        "font_family": "Fraunces",
        "bullets": ["one", "two", "three"],
    },
    "puzzle": {
        "type": "maze",
        "algorithm": "prim",
        "count": 12,
        "difficulty_curve": ["easy", "medium"],
        "grid_sizes": {"easy": [8, 8], "medium": [10, 10], "hard": [12, 12]},
        "include_solutions": True,
        "solutions_section": "end",
    },
    "front_matter": {"title_page": True, "intro_page": False, "intro_text": ""},
}


def _settings(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(output_dir=tmp_path)


@pytest.fixture
def patched_orchestrator(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, make_book):
    """Wire up the orchestrator with mocked DB + assemblers + provider seams.

    The orchestrator's behaviour under test is purely its phase/gate logic;
    the actual assemblers, providers, and DB calls are mocked. Returns a
    SimpleNamespace of the mocks for assertions.
    """
    from src.config.schema import NicheConfig

    config = NicheConfig.model_validate(_PUZZLE_YAML_DICT)

    state = {"status": BookStatus.QA_DONE}  # entering at QA_DONE — Gate 2 next.

    def _resolve(_target: str) -> tuple[Any, Any]:
        # Re-fetch the book with whatever status the test has progressed it to.
        return (
            make_book(slug=config.slug, status=state["status"]),
            config,
        )

    def _transition(_id: Any, status: BookStatus, *_args: Any, **_kw: Any) -> None:
        state["status"] = status

    def _get_by_id(_id: Any) -> Any:
        return make_book(slug=config.slug, status=state["status"])

    assemble_interior = Mock(return_value=tmp_path / "interior.pdf")
    assemble_cover = Mock(return_value=tmp_path / "cover.pdf")
    run_metadata = Mock()
    build_summary = Mock()

    monkeypatch.setattr(orchestrator, "resolve_book", _resolve)
    monkeypatch.setattr(orchestrator, "transition_status", _transition)
    monkeypatch.setattr(orchestrator, "get_book_by_id", _get_by_id)
    monkeypatch.setattr(orchestrator, "_assemble_puzzle_interior", assemble_interior)
    monkeypatch.setattr(orchestrator, "_assemble_puzzle_cover", assemble_cover)
    monkeypatch.setattr(orchestrator, "run_metadata", run_metadata)
    monkeypatch.setattr(orchestrator, "_build_summary", build_summary)
    monkeypatch.setattr(orchestrator, "get_settings", lambda: _settings(tmp_path))
    monkeypatch.setattr(
        orchestrator, "get_anthropic_provider", lambda: Mock(name="AnthropicProvider")
    )
    # book_log_file is a context manager — no-op for tests.
    from contextlib import nullcontext

    monkeypatch.setattr(orchestrator, "book_log_file", lambda *_a, **_kw: nullcontext())

    return SimpleNamespace(
        state=state,
        assemble_interior=assemble_interior,
        assemble_cover=assemble_cover,
        run_metadata=run_metadata,
        build_summary=build_summary,
    )


def test_orchestrator_pauses_after_assembly_without_yes(
    patched_orchestrator, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Guardrail #7: assemble interior + cover, print the gate message, then STOP.

    Without --yes, the orchestrator must NOT call run_metadata. The book
    stays at status ASSEMBLING so --resume can continue it.
    """
    orchestrator.run_puzzle_build("dummy.yaml", assume_yes=False, resume=True)

    p = patched_orchestrator
    assert p.assemble_interior.call_count == 1
    assert p.assemble_cover.call_count == 1
    # The gate must keep metadata generation from running.
    p.run_metadata.assert_not_called()
    p.build_summary.assert_not_called()
    out = capsys.readouterr().out
    assert "Review interior.pdf and cover.pdf" in out
    assert "puzzle build --resume" in out
    # And the book status sits at ASSEMBLING — Phase E waits for --resume.
    assert p.state["status"] == BookStatus.ASSEMBLING


def test_orchestrator_with_yes_skips_gate_and_runs_metadata(
    patched_orchestrator, tmp_path: Path
) -> None:
    """--yes skips the review pause and continues straight into metadata generation."""
    orchestrator.run_puzzle_build("dummy.yaml", assume_yes=True, resume=False)

    p = patched_orchestrator
    p.assemble_interior.assert_called_once()
    p.assemble_cover.assert_called_once()
    p.run_metadata.assert_called_once()
    p.build_summary.assert_called_once()
