"""Evaluate a selected checkpoint without modifying its training artifacts."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from relstream_ode.trainer import evaluate_main

if __name__ == "__main__":
    evaluate_main()
