"""Compatibility entry point for textbook benchmark baselines."""

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.storm_examples.run_storm_textbook_benchmark import main


if __name__ == "__main__":
    raise SystemExit(main())
