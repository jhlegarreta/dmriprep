#!/usr/bin/env python
"""Download DIPY datasets required for the test suite."""

from __future__ import annotations

import os
from pathlib import Path

from dmriprep.data.tests import testdata_fetcher


def main() -> None:
    """Fetch the Sherbrooke 3-shell dataset into the cache directory."""
    dipy_datadir_root = Path(os.getenv('DMRIPREP_TESTS_DATA') or Path.home())
    dipy_datadir = dipy_datadir_root / '.cache' / 'data'
    dipy_datadir.mkdir(parents=True, exist_ok=True)

    testdata_fetcher(folder=str(dipy_datadir))()


if __name__ == '__main__':
    main()
