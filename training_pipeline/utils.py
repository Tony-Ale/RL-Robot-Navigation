import json
import os
import random
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REWARD_CONFIG_PATH = PROJECT_ROOT / "custom_rewards" / "social_safety_reward_config.yaml"
REWARD_CONFIG_SNAPSHOT_NAME = "social_safety_reward_config.yaml"


def configure_matplotlib_cache() -> None:
    """Use a writable Matplotlib cache directory unless the user set one."""
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")


def load_yaml(path: str) -> Dict[str, Any]:
    """Load a YAML file into a dictionary."""
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required for the training pipeline.") from exc

    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def save_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=4, sort_keys=True, default=str)


def record_training_seed_session(run_dir: Path, config: Dict[str, Any]) -> Dict[str, Any]:
    """Append the training seed used for this launch to the run history."""
    path = Path(run_dir) / "training_seed_history.json"
    if path.exists() and path.stat().st_size > 0:
        with open(path, "r") as f:
            data = json.load(f)
    else:
        data = {"sessions": []}

    experiment = config["experiment"]
    training = config["training"]
    resume_checkpoint = training.get("resume_from_checkpoint")
    session = {
        "session": len(data["sessions"]) + 1,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "mode": "resume" if resume_checkpoint else "initial",
        "training_seed": int(experiment["seed"]),
        "resume_run_dir": experiment.get("resume_run_dir"),
        "resume_from_checkpoint": resume_checkpoint,
        "reset_num_timesteps": training.get("reset_num_timesteps", True),
        "total_timesteps": training.get("total_timesteps"),
    }
    data["sessions"].append(session)
    save_json(path, data)
    return data


def copy_reward_config_once(run_dir: Path) -> None:
    """Snapshot the reward parameters without overwriting an existing run copy."""
    destination = Path(run_dir) / REWARD_CONFIG_SNAPSHOT_NAME
    if not destination.exists():
        shutil.copy2(REWARD_CONFIG_PATH, destination)


def make_run_dir(config: Dict[str, Any], config_path: Optional[str] = None) -> Path:
    """Create and return the folder for a single experiment run."""
    experiment = config["experiment"]
    if experiment.get("resume_run_dir"):
        run_dir = Path(experiment["resume_run_dir"])
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "checkpoints").mkdir(exist_ok=True)
        (run_dir / "metrics").mkdir(exist_ok=True)
        (run_dir / "tensorboard").mkdir(exist_ok=True)
        if config_path is not None and experiment.get("copy_config", True):
            shutil.copy2(config_path, run_dir / "latest_experiment_config.yaml")
            copy_reward_config_once(run_dir)
        return run_dir

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = experiment.get("run_id") or f"{timestamp}_{experiment['name']}"
    run_dir = Path(experiment["output_root"]) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "checkpoints").mkdir()
    (run_dir / "metrics").mkdir()
    (run_dir / "tensorboard").mkdir()

    if config_path is not None and experiment.get("copy_config", True):
        shutil.copy2(config_path, run_dir / "experiment_config.yaml")
        copy_reward_config_once(run_dir)
    save_json(run_dir / "metadata.json", {"created_at": timestamp, "run_id": run_id})
    return run_dir


def set_global_seeds(seed: int) -> None:
    """Set common Python, NumPy, and PyTorch seeds."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
