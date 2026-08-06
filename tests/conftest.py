"""Ensures the repo root is importable as `core`, `modules`, and `main` during tests."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
