"""Run Model Doctor as a Python module."""

import sys

from .cli import main


if __name__ == "__main__":
    sys.exit(
        main()
    )