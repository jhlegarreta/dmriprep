"""Test data helpers."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Sequence
from functools import partial

from dipy.data.fetcher import UW_RW_URL, _make_fetcher

testdata_fetcher = partial(
    _make_fetcher,
    name='fetch_sherbrooke_3shell',
    baseurl=UW_RW_URL + '1773/38475/',
    remote_fnames=['HARDI193.nii.gz', 'HARDI193.bval', 'HARDI193.bvec'],
    local_fnames=['HARDI193.nii.gz', 'HARDI193.bval', 'HARDI193.bvec'],
    md5_list=[
        '0b735e8f16695a37bfbd66aab136eb66',
        'e9b9bb56252503ea49d31fb30a0ac637',
        '0c83f7e8b917cd677ad58a078658ebb7',
    ],
    doc='Download a 3shell HARDI dataset with 192 gradient direction',
)

__all__ = ['testdata_fetcher', 'main']


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


if __name__ == '__main__':  # pragma: no cover - convenience entrypoint
    main()
