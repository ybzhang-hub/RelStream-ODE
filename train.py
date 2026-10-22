"""Train the main model or a component ablation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from relstream_ode.trainer import main

if __name__ == "__main__":
    main()
