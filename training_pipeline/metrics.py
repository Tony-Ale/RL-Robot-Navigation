import csv
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np


NAVIGATION_METRIC_KEYS = [
    "SUCCESS",
    "COLLISION",
    "COLLISION_HUMAN",
    "COLLISION_OBJECT",
    "COLLISION_WALL",
    "OUT_OF_MAP",
    "TIMEOUT",
    "TIME_TO_REACH_GOAL",
    "PATH_LENGTH",
    "A_STAR_PATH_LENGTH",
    "A_STAR_SPL",
    "SPL",
    "STL",
    "MINIMUM_DISTANCE_TO_HUMAN",
    "TIME_TO_COLLISION",
    "PERSONAL_SPACE_COMPLIANCE",
    "MINIMUM_OBSTACLE_DISTANCE",
    "AVERAGE_OBSTACLE_DISTANCE",
    "V_MIN",
    "V_AVG",
    "V_MAX",
    "A_MIN",
    "A_AVG",
    "A_MAX",
    "JERK_MIN",
    "JERK_AVG",
    "JERK_MAX",
    "FAILURE_TO_PROGRESS",
    "STALLED_TIME",
]


class CSVMetricWriter:
    """Append metric rows to a CSV file with stable headers."""

    def __init__(self, path: Path, fieldnames: Iterable[str]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames = list(fieldnames)
        self._wrote_header = self.path.exists() and self.path.stat().st_size > 0

    def write(self, row: Dict[str, Any]) -> None:
        with open(self.path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames, extrasaction="ignore")
            if not self._wrote_header:
                writer.writeheader()
                self._wrote_header = True
            writer.writerow({key: clean_csv_value(row.get(key)) for key in self.fieldnames})


def clean_csv_value(value):
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        if np.isinf(value) or np.isnan(value):
            return ""
        return float(value)
    if value is None:
        return ""
    if isinstance(value, (str, bytes)):
        return value
    return str(value)


def count_existing_rows(path: Path) -> int:
    """Count existing CSV data rows so resumed runs continue numbering."""
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return 0
    with open(path, "r", newline="") as f:
        return max(sum(1 for _ in f) - 1, 0)


def numeric_values(rows: Iterable[Dict[str, Any]], key: str) -> List[float]:
    values = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, (int, float, bool, np.integer, np.floating, np.bool_)):
            if not np.isnan(float(value)) and not np.isinf(float(value)):
                values.append(float(value))
    return values


def mean_or_none(values: Iterable[float]):
    values = list(values)
    if not values:
        return None
    return float(np.mean(values))
