"""Convert native ICEWS or GDELT 2.0 event exports."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from relstream_ode.native_stream import main

if __name__ == "__main__":
    main()
