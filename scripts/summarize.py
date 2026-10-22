"""Export actual per-seed metrics and matched confidence intervals."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from relstream_ode.statistics import main

if __name__ == "__main__":
    main()
