"""Small output helpers shared by environment-inspection analyses."""

import csv
from pathlib import Path
from typing import Mapping, Sequence


def outcome_from_info(info: Mapping, fallback: str = "other") -> str:
    """Classify an episode using the terminal metrics emitted by the environment."""
    if info.get("SUCCESS"):
        return "success"
    if info.get("COLLISION"):
        return "collision"
    if info.get("TIMEOUT"):
        return "timeout"
    return fallback


def write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping]) -> None:
    """Write analysis rows with a stable column order, ignoring internal fields."""
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
