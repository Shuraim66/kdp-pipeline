"""Tests for the CLI surface — Phase 10 orchestration and maintenance commands.

The provider/DB-touching happy paths (`build`'s generation phases, …) are
exercised end-to-end against live services. Here we cover everything reachable
without a network round-trip: the command tree, `--help`/`--version`, the pure
`find_orphan_images` helper, and every command's argument-handling and
error-branch logic with the DB/provider seams monkeypatched.
"""

from __future__ import annotations

from collections.abc import Callable
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from click.testing import CliRunner, Result
from src.db.models import BookStatus, IllegalTransitionError, StatusLogEntry
from src.main import _ink_density_band, cli, find_orphan_images

_NICHES_DIR = Path(__file__).resolve().parent.parent / "niches"
_NURSES_YAML = str(_NICHES_DIR / "nurses_v1.yaml")

# Every subcommand the CLI must expose (spec Phase 10).
_EXPECTED_COMMANDS = {
    "init-db",
    "db-status",
    "validate-niche",
    "generate-images",
    "run-qa",
    "review-qa",
    "build-interior",
    "build-cover",
    "generate-metadata",
    "check-env",
    "build",
    "list-books",
    "show",
    "show-costs",
    "set-asin",
    "mark-published",
    "retry-failed",
    "cleanup-orphans",
}


def _text(result: Result) -> str:
    """Combined stdout+stderr — click 8.2+ keeps the streams separate."""
    return result.output + (result.stderr or "")


def _settings(tmp_path: Path, **overrides: Any) -> SimpleNamespace:
    """A stand-in for `Settings` exposing just the attributes the CLI reads."""
    base: dict[str, Any] = {
        "database_url": "postgresql://u:p@host:5432/postgres",
        "anthropic_api_key": "sk-ant-test",
        "fal_key": "fal-test",
        "output_dir": tmp_path,
        "max_book_cost_usd": None,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


class _StubCursor:
    def execute(self, *args: Any, **kwargs: Any) -> None:
        pass

    def __enter__(self) -> _StubCursor:
        return self

    def __exit__(self, *args: Any) -> bool:
        return False


class _StubConn:
    def cursor(self, *args: Any, **kwargs: Any) -> _StubCursor:
        return _StubCursor()

    def __enter__(self) -> _StubConn:
        return self

    def __exit__(self, *args: Any) -> bool:
        return False


class _StubPool:
    def __init__(self, *, fail: bool = False) -> None:
        self._fail = fail

    def connection(self) -> _StubConn:
        if self._fail:
            raise RuntimeError("password authentication failed for user")
        return _StubConn()


# --------------------------------------------------------------------------
# command tree
# --------------------------------------------------------------------------


def test_all_commands_registered() -> None:
    assert set(cli.commands) >= _EXPECTED_COMMANDS


def test_cli_help_lists_every_command() -> None:
    result = CliRunner().invoke(cli, ["--help"])
    assert result.exit_code == 0
    for name in _EXPECTED_COMMANDS:
        assert name in result.output


def test_cli_version() -> None:
    result = CliRunner().invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.output


def test_build_help_lists_its_flags() -> None:
    result = CliRunner().invoke(cli, ["build", "--help"])
    assert result.exit_code == 0
    for flag in ("--yes", "--resume", "--test-images"):
        assert flag in result.output


def test_build_rejects_missing_yaml() -> None:
    result = CliRunner().invoke(cli, ["build", "does_not_exist.yaml"])
    assert result.exit_code != 0


# --------------------------------------------------------------------------
# find_orphan_images
# --------------------------------------------------------------------------


def test_find_orphan_images_missing_dir(tmp_path: Path) -> None:
    assert find_orphan_images(tmp_path / "absent", set()) == []


def test_find_orphan_images_returns_unreferenced(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    referenced_a = raw / "001.png"
    orphan = raw / "002.png"
    referenced_c = raw / "003.png"
    for path in (referenced_a, orphan, referenced_c):
        path.write_bytes(b"png")

    orphans = find_orphan_images(raw, {referenced_a.resolve(), referenced_c.resolve()})

    assert orphans == [orphan]


def test_find_orphan_images_is_sorted(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    for name in ("003.png", "001.png", "002.png"):
        (raw / name).write_bytes(b"png")

    orphans = find_orphan_images(raw, set())

    assert [p.name for p in orphans] == ["001.png", "002.png", "003.png"]


def test_find_orphan_images_ignores_non_png(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "001.png").write_bytes(b"png")
    (raw / "failed_prompts.txt").write_bytes(b"text")

    assert [p.name for p in find_orphan_images(raw, set())] == ["001.png"]


def test_find_orphan_images_all_referenced(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    kept = raw / "001.png"
    kept.write_bytes(b"png")

    assert find_orphan_images(raw, {kept.resolve()}) == []


# --------------------------------------------------------------------------
# _ink_density_band — reads the advisory band defensively from a stored config
# --------------------------------------------------------------------------


def test_ink_density_band_reads_from_config() -> None:
    assert _ink_density_band({"qa": {"ink_density_band": [4.0, 9.0]}}) == (4.0, 9.0)


def test_ink_density_band_defaults_when_absent() -> None:
    # A book whose stored config predates the field falls back to the default.
    assert _ink_density_band({"qa": {"min_white_pct": 90.0}}) == (3.0, 8.0)


def test_ink_density_band_defaults_on_stale_config() -> None:
    # A pre-Q4 config (no `qa` section at all) must not raise — just default.
    assert _ink_density_band({}) == (3.0, 8.0)


# --------------------------------------------------------------------------
# validate-niche
# --------------------------------------------------------------------------


def test_validate_niche_valid() -> None:
    result = CliRunner().invoke(cli, ["validate-niche", _NURSES_YAML])
    assert result.exit_code == 0
    assert "VALID" in result.output
    assert "config_hash" in result.output


def test_validate_niche_missing_file() -> None:
    result = CliRunner().invoke(cli, ["validate-niche", "nope.yaml"])
    assert result.exit_code != 0


# --------------------------------------------------------------------------
# check-env
# --------------------------------------------------------------------------


def test_check_env_all_good(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("src.main.get_settings", lambda: _settings(tmp_path))
    monkeypatch.setattr("src.main.get_pool", lambda: _StubPool())
    result = CliRunner().invoke(cli, ["check-env"])
    assert result.exit_code == 0
    assert "Environment OK" in result.output


def test_check_env_missing_keys(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "src.main.get_settings",
        lambda: _settings(tmp_path, anthropic_api_key=None, fal_key=None),
    )
    monkeypatch.setattr("src.main.get_pool", lambda: _StubPool())
    result = CliRunner().invoke(cli, ["check-env"])
    assert result.exit_code == 1
    assert "ANTHROPIC_API_KEY" in _text(result)


def test_check_env_wrong_port(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "src.main.get_settings",
        lambda: _settings(tmp_path, database_url="postgresql://u:p@host:6543/postgres"),
    )
    monkeypatch.setattr("src.main.get_pool", lambda: _StubPool())
    result = CliRunner().invoke(cli, ["check-env"])
    assert result.exit_code == 1
    assert "6543" in _text(result)


def test_check_env_db_unreachable(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("src.main.get_settings", lambda: _settings(tmp_path))
    monkeypatch.setattr("src.main.get_pool", lambda: _StubPool(fail=True))
    result = CliRunner().invoke(cli, ["check-env"])
    assert result.exit_code == 1
    assert "authentication failed" in _text(result)


# --------------------------------------------------------------------------
# list-books / show
# --------------------------------------------------------------------------


def test_list_books_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("src.main.list_books", lambda status=None, niche=None: [])
    result = CliRunner().invoke(cli, ["list-books"])
    assert result.exit_code == 0
    assert "No books match" in result.output


def test_list_books_some(monkeypatch: pytest.MonkeyPatch, make_book: Callable[..., Any]) -> None:
    book = make_book(title="Nurse Life", status=BookStatus.READY)
    monkeypatch.setattr("src.main.list_books", lambda status=None, niche=None: [book])
    result = CliRunner().invoke(cli, ["list-books"])
    assert result.exit_code == 0
    assert book.slug in result.output
    assert "1 book(s)" in result.output


def test_show_unknown_slug(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("src.main.get_book_by_slug", lambda slug: None)
    result = CliRunner().invoke(cli, ["show", "ghost"])
    assert result.exit_code == 1
    assert "no book" in _text(result)


def test_show_existing(monkeypatch: pytest.MonkeyPatch, make_book: Callable[..., Any]) -> None:
    book = make_book(title="Nurse Life", status=BookStatus.READY)
    entry = StatusLogEntry(
        id=book.id,
        book_id=book.id,
        from_status=BookStatus.CREATED,
        to_status=BookStatus.GENERATING,
        reason=None,
        created_at=book.created_at,
    )
    monkeypatch.setattr("src.main.get_book_by_slug", lambda slug: book)
    monkeypatch.setattr("src.main.get_status_log", lambda book_id: [entry])
    monkeypatch.setattr("src.main.total_cost_for_book", lambda book_id: Decimal("1.23"))
    result = CliRunner().invoke(cli, ["show", book.slug])
    assert result.exit_code == 0
    assert "Nurse Life" in result.output
    assert "1.23" in result.output


# --------------------------------------------------------------------------
# show-costs
# --------------------------------------------------------------------------


def test_show_costs_all(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("src.main.total_cost_across_books", lambda date_range: Decimal("4.50"))
    result = CliRunner().invoke(cli, ["show-costs", "--all"])
    assert result.exit_code == 0
    assert "4.50" in result.output


def test_show_costs_all_since(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def _fake(date_range: Any) -> Decimal:
        captured["range"] = date_range
        return Decimal("2.00")

    monkeypatch.setattr("src.main.total_cost_across_books", _fake)
    result = CliRunner().invoke(cli, ["show-costs", "--all", "--since", "2026-01-01"])
    assert result.exit_code == 0
    assert captured["range"] is not None


def test_show_costs_bad_since(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("src.main.total_cost_across_books", lambda date_range: Decimal(0))
    result = CliRunner().invoke(cli, ["show-costs", "--all", "--since", "yesterday"])
    assert result.exit_code == 1
    assert "YYYY-MM-DD" in _text(result)


def test_show_costs_needs_target() -> None:
    result = CliRunner().invoke(cli, ["show-costs"])
    assert result.exit_code == 1
    assert "--all" in _text(result)


def test_show_costs_per_book(
    monkeypatch: pytest.MonkeyPatch, make_book: Callable[..., Any]
) -> None:
    book = make_book()
    monkeypatch.setattr("src.main.get_book_by_slug", lambda slug: book)
    monkeypatch.setattr(
        "src.main.cost_breakdown_for_book",
        lambda book_id: {"fal": Decimal("0.42"), "anthropic": Decimal("0.03")},
    )
    monkeypatch.setattr("src.main.total_cost_for_book", lambda book_id: Decimal("0.45"))
    result = CliRunner().invoke(cli, ["show-costs", book.slug])
    assert result.exit_code == 0
    assert "fal" in result.output
    assert "0.45" in result.output


# --------------------------------------------------------------------------
# set-asin / mark-published / retry-failed
# --------------------------------------------------------------------------


def test_set_asin(monkeypatch: pytest.MonkeyPatch, make_book: Callable[..., Any]) -> None:
    book = make_book()
    recorded: dict[str, Any] = {}
    monkeypatch.setattr("src.main.get_book_by_slug", lambda slug: book)
    monkeypatch.setattr(
        "src.main.set_asin",
        lambda book_id, asin: recorded.update(book_id=book_id, asin=asin),
    )
    result = CliRunner().invoke(cli, ["set-asin", book.slug, "B0ABCD1234"])
    assert result.exit_code == 0
    assert recorded["asin"] == "B0ABCD1234"


def test_set_asin_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("src.main.get_book_by_slug", lambda slug: None)
    result = CliRunner().invoke(cli, ["set-asin", "ghost", "B0ABCD1234"])
    assert result.exit_code == 1


def test_mark_published(monkeypatch: pytest.MonkeyPatch, make_book: Callable[..., Any]) -> None:
    book = make_book(status=BookStatus.READY)
    monkeypatch.setattr("src.main.get_book_by_slug", lambda slug: book)
    monkeypatch.setattr("src.main.transition_status", lambda book_id, to: None)
    result = CliRunner().invoke(cli, ["mark-published", book.slug])
    assert result.exit_code == 0
    assert "published" in result.output


def test_mark_published_illegal(
    monkeypatch: pytest.MonkeyPatch, make_book: Callable[..., Any]
) -> None:
    book = make_book(status=BookStatus.CREATED)

    def _raise(book_id: Any, to: Any) -> None:
        raise IllegalTransitionError(BookStatus.CREATED, BookStatus.PUBLISHED)

    monkeypatch.setattr("src.main.get_book_by_slug", lambda slug: book)
    monkeypatch.setattr("src.main.transition_status", _raise)
    result = CliRunner().invoke(cli, ["mark-published", book.slug])
    assert result.exit_code == 1
    assert "ready" in _text(result)


def test_retry_failed_resets(
    monkeypatch: pytest.MonkeyPatch, make_book: Callable[..., Any]
) -> None:
    book = make_book(status=BookStatus.FAILED)
    monkeypatch.setattr("src.main.get_book_by_slug", lambda slug: book)
    monkeypatch.setattr("src.main.reset_failed_book", lambda book_id: True)
    result = CliRunner().invoke(cli, ["retry-failed", book.slug])
    assert result.exit_code == 0
    assert "reset" in result.output


def test_retry_failed_not_failed(
    monkeypatch: pytest.MonkeyPatch, make_book: Callable[..., Any]
) -> None:
    book = make_book(status=BookStatus.READY)
    monkeypatch.setattr("src.main.get_book_by_slug", lambda slug: book)
    monkeypatch.setattr("src.main.reset_failed_book", lambda book_id: False)
    result = CliRunner().invoke(cli, ["retry-failed", book.slug])
    assert result.exit_code == 0
    assert "nothing to reset" in result.output


# --------------------------------------------------------------------------
# cleanup-orphans
# --------------------------------------------------------------------------


def test_cleanup_orphans_none(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, make_book: Callable[..., Any]
) -> None:
    book = make_book()
    monkeypatch.setattr("src.main.get_book_by_slug", lambda slug: book)
    monkeypatch.setattr("src.main.list_images", lambda book_id: [])
    monkeypatch.setattr("src.main.get_settings", lambda: _settings(tmp_path))
    result = CliRunner().invoke(cli, ["cleanup-orphans", book.slug])
    assert result.exit_code == 0
    assert "no orphan" in result.output


def test_cleanup_orphans_deletes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, make_book: Callable[..., Any]
) -> None:
    book = make_book()
    raw = tmp_path / book.slug / "images" / "raw"
    raw.mkdir(parents=True)
    orphan = raw / "001.png"
    orphan.write_bytes(b"png")
    monkeypatch.setattr("src.main.get_book_by_slug", lambda slug: book)
    monkeypatch.setattr("src.main.list_images", lambda book_id: [])
    monkeypatch.setattr("src.main.get_settings", lambda: _settings(tmp_path))
    result = CliRunner().invoke(cli, ["cleanup-orphans", book.slug, "--yes"])
    assert result.exit_code == 0
    assert not orphan.exists()


def test_cleanup_orphans_keeps_referenced(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    make_book: Callable[..., Any],
    make_image: Callable[..., Any],
) -> None:
    book = make_book()
    raw = tmp_path / book.slug / "images" / "raw"
    raw.mkdir(parents=True)
    kept = raw / "001.png"
    kept.write_bytes(b"png")
    image = make_image(book_id=book.id, local_path=str(kept))
    monkeypatch.setattr("src.main.get_book_by_slug", lambda slug: book)
    monkeypatch.setattr("src.main.list_images", lambda book_id: [image])
    monkeypatch.setattr("src.main.get_settings", lambda: _settings(tmp_path))
    result = CliRunner().invoke(cli, ["cleanup-orphans", book.slug, "--yes"])
    assert result.exit_code == 0
    assert kept.exists()


# --------------------------------------------------------------------------
# build — guard branches (phases need live providers, tested e2e)
# --------------------------------------------------------------------------


def test_build_resolve_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def _raise(target: str) -> Any:
        raise ValueError("bad config")

    monkeypatch.setattr("src.main.resolve_book", _raise)
    monkeypatch.setattr("src.main.get_settings", lambda: _settings(tmp_path))
    result = CliRunner().invoke(cli, ["build", _NURSES_YAML])
    assert result.exit_code == 1
    assert "bad config" in _text(result)


def test_build_failed_book(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    make_book: Callable[..., Any],
    niche_config: Any,
) -> None:
    book = make_book(status=BookStatus.FAILED, failure_phase="generation", failure_reason="boom")
    monkeypatch.setattr("src.main.resolve_book", lambda target: (book, niche_config))
    monkeypatch.setattr("src.main.get_settings", lambda: _settings(tmp_path))
    result = CliRunner().invoke(cli, ["build", _NURSES_YAML])
    assert result.exit_code == 1
    assert "retry-failed" in _text(result)


def test_build_mid_pipeline_needs_resume(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    make_book: Callable[..., Any],
    niche_config: Any,
) -> None:
    book = make_book(status=BookStatus.GENERATION_DONE)
    monkeypatch.setattr("src.main.resolve_book", lambda target: (book, niche_config))
    monkeypatch.setattr("src.main.get_settings", lambda: _settings(tmp_path))
    result = CliRunner().invoke(cli, ["build", _NURSES_YAML])
    assert result.exit_code == 1
    assert "--resume" in _text(result)


def test_build_already_ready_prints_summary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    make_book: Callable[..., Any],
    niche_config: Any,
) -> None:
    book = make_book(status=BookStatus.READY, title="Nurse Life")
    monkeypatch.setattr("src.main.resolve_book", lambda target: (book, niche_config))
    monkeypatch.setattr("src.main.get_settings", lambda: _settings(tmp_path))
    monkeypatch.setattr("src.main.get_book_by_id", lambda book_id: book)
    monkeypatch.setattr("src.main.cost_breakdown_for_book", lambda book_id: {})
    monkeypatch.setattr("src.main.total_cost_for_book", lambda book_id: Decimal(0))
    result = CliRunner().invoke(cli, ["build", _NURSES_YAML])
    assert result.exit_code == 0
    assert "already ready" in result.output
    assert "Book ready" in result.output
