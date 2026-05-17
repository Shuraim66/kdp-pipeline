# kdp_pipeline

Production CLI pipeline that turns a niche config into a print-ready Amazon KDP
coloring book: interior PDF, cover PDF, and listing metadata.

## Stack

Python 3.11+ (managed with [uv](https://docs.astral.sh/uv/)) · Supabase Postgres
(`psycopg` 3) · Fal.ai image generation · Anthropic metadata ·
`reportlab`/`pypdf` PDF assembly · `click` CLI.

## Setup

    uv sync
    cp .env.example .env   # fill in real values

## Usage

    uv run python -m src.main --help

## Development

    uv run mypy src/         # strict type check
    uv run ruff check src/   # lint
    uv run ruff format src/  # format
