"""manages user-specific profile data for adaptive scoring and thresholds"""

import json
import os

import numpy as np

class RunningStat:
    """maintains running statistics to be used for adaptive/personalized scoring and thresholds"""
    def __init__(self, mean=0.0, var=1.0, count=0.0):
        self.mean = float(mean)
        self.m2 = float(max(var, 1e-6) * max(count, 1.0))
        self.count = float(count)

    def update(self, x: float):
        """update profile with new data"""
        self.count += 1.0
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.m2 += delta * delta2

    @property
    def var(self):
        """return current variance (with small epsilon to prevent division issues)"""
        if self.count <= 1:
            return 1e-4
        return max(self.m2 / (self.count - 1.0), 1e-4)

    @property
    def std(self):
        """return standard deviation"""
        return float(np.sqrt(self.var))

    def to_dict(self):
        """convert to dict for JSON storage"""
        return {"mean": self.mean, "var": self.var, "count": self.count}

    @classmethod
    def from_dict(cls, data, default_mean, default_var):
        """create RunningStat from dict, with defaults if data is missing or invalid"""
        if not data:
            return cls(default_mean, default_var, 0.0)

        return cls(
            data.get("mean", default_mean), data.get("var", default_var), data.get("count", 0.0)
        )


class DriverProfile:
    """manages user-specific statistics for adaptive scoring and thresholds"""
    # defaults for new users or missing data
    DEFAULTS = {
        "ear": (0.28, 0.0025),
        "mar": (0.16, 0.0020),
        "blink_rate": (15.0, 16.0),
        "roll_deg": (0.0, 20.0),
        "yaw_ratio": (0.0, 0.01),
        "pitch_ratio": (0.58, 0.01),
    }

    def __init__(self, path: str):
        self.path = path
        self.stats = {}
        self.sessions = 0
        self.total_updates = 0
        self._load()

    def save(self):
        """save profile to disk"""
        parent = os.path.dirname(self.path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump({
                "sessions": self.sessions,
                "total_updates": self.total_updates,
                "stats": {k: v.to_dict() for k, v in self.stats.items()}
            }, f, indent=2)

    def _load(self):
        """load data from disk (profile or config)"""
        payload = {}
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
            except (OSError, json.JSONDecodeError):
                payload = {}

        self.sessions = int(payload.get("sessions", 0))
        self.total_updates = int(payload.get("total_updates", 0))
        saved = payload.get("stats", {})

        for key, (default_mean, default_var) in self.DEFAULTS.items():
            self.stats[key] = RunningStat.from_dict(saved.get(key), default_mean, default_var)

    def reset_baseline(self):
        """reset baseline completely (for new user / recalibration)"""
        self.total_updates = 0
        self.stats = {}

        for key, (mean, var) in self.DEFAULTS.items():
            self.stats[key] = RunningStat(mean, var, 0)

    def mean(self, key: str) -> float:
        """get current mean for a given feature"""
        return self.stats[key].mean

    def std(self, key: str) -> float:
        """get current standard deviation for a given feature"""
        return self.stats[key].std

    def update_from_alert_frame(self, features: dict):
        """update profile with new data only from alert frames."""
        for key in self.DEFAULTS:
            self.stats[key].update(float(features[key]))

        self.total_updates += 1

    def begin_session(self):
        """increment session count"""
        self.sessions += 1
