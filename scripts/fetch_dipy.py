#!/usr/bin/env python
"""Download DIPY datasets required for the test suite."""

from __future__ import annotations

from dmriprep.data.tests import main as fetch_main


def main() -> None:
    """Fetch the Sherbrooke 3-shell dataset into the cache directory."""
    fetch_main([])


if __name__ == '__main__':
    main()
