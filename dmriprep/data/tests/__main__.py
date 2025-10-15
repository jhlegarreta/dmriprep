# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright The NiPreps Developers <nipreps@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# We support and encourage derived works from this project, please read
# about our expectations at
#
#     https://www.nipreps.org/community/licensing/
#
"""Module entrypoint for ``python -m dmriprep.data.tests``."""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from pathlib import Path

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
