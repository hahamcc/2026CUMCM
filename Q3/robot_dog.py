"""Direct launch script for the Problem 3 robot dog.

Run from the project root, for example:
    python Q3/robot_dog.py --robot-id "<current simulator team id>"
"""

from __future__ import annotations

import sys
from pathlib import Path


if __package__ in {None, ""}:
    # Allow direct execution while retaining the package's relative imports.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from Q3.run import main


if __name__ == "__main__":
    raise SystemExit(main())

