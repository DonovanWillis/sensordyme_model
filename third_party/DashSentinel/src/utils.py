"""
Contains some basic utility functions I'll use across the project.
Could be replaced with a more robust library.
"""

import time
from pathlib import Path

import numpy as np


def clamp(value, low, high):
    """clamp between high and low"""
    return max(low, min(high, value))

def euclidean(p1, p2):
    """calculates the euclidean distance between two points"""
    return float(np.linalg.norm(np.array(p1) - np.array(p2)))


def ensure_parent(path: str):
    """ensures that the parent directory of the given path exists, creating it if necessary"""
    Path(path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)

def now_ts() -> str:
    """Get the current timestamp as a formatted string"""
    return time.strftime("%Y-%m-%d %H:%M:%S")
