"""kdp_pipeline CLI entrypoint.

Invoked as ``python -m src.main``. Subcommands are attached to the ``cli``
group in later phases (check-env, db-status, build, ...).
"""

import click


@click.group()
@click.version_option("0.1.0", prog_name="kdp_pipeline")
def cli() -> None:
    """KDP coloring book pipeline."""


if __name__ == "__main__":
    cli()
