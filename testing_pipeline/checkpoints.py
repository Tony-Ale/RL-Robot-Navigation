"""Checkpoint naming and resolution helpers for evaluation tools."""

from pathlib import Path


def checkpoint_step(path: Path) -> int:
    """Extract the trailing training step from a pipeline checkpoint filename."""
    try:
        return int(Path(path).stem.rsplit("_", 1)[-1])
    except ValueError:
        return -1
