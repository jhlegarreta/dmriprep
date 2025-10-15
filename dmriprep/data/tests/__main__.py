"""Module entrypoint for ``python -m dmriprep.data.tests``."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Sequence

from . import testdata_fetcher

__all__ = ['main']


def _default_destination() -> Path:
    """Return the default directory for storing the test dataset."""

    root = Path(os.getenv('DMRIPREP_TESTS_DATA') or Path.home())
    return root / '.cache' / 'data'


def main(argv: Sequence[str] | None = None) -> None:
    """Fetch the Sherbrooke 3-shell dataset for use in tests."""

    parser = argparse.ArgumentParser(
        description='Download the Sherbrooke 3-shell dataset required by the tests.',
    )
    parser.add_argument(
        'destination',
        nargs='?',
        help=(
            'Directory to store the dataset. Defaults to '
            '${DMRIPREP_TESTS_DATA:-$HOME}/.cache/data.'
        ),
    )
    args = parser.parse_args(argv)

    destination = Path(args.destination) if args.destination else _default_destination()
    destination.mkdir(parents=True, exist_ok=True)

    testdata_fetcher(folder=str(destination))()


if __name__ == '__main__':  # pragma: no cover - module entrypoint
    main()
