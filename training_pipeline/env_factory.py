import sys
from pathlib import Path
from typing import Any, Dict

from training_pipeline.utils import configure_matplotlib_cache, load_yaml

configure_matplotlib_cache()

import gym

from training_pipeline.action_wrappers import (
    DifferentialDriveActionWrapper,
    DropEmptyObservationKeysWrapper,
    FixedObservationSpaceWrapper,
    GymnasiumCompatibilityWrapper,
)
from training_pipeline.observation_history_wrapper import ObservationHistoryWrapper


def make_socnav_env(config: Dict[str, Any], rank: int = 0, eval_mode: bool = False):
    """Build one configured SocNavGym environment instance."""
    env_cfg = config["environment"]
    seed = int(config["experiment"]["seed"]) + rank
    if eval_mode:
        seed = int(config["evaluation"]["eval_seed_base"]) + rank

    if env_cfg.get("use_socnavgym_clone", False):
        clone_path = str(Path(env_cfg["socnavgym_clone_path"]).resolve())
        if clone_path not in sys.path:
            sys.path.insert(0, clone_path)

    wrappers = config.get("wrappers", {})
    fixed_cfg = wrappers.get("fixed_observation_space", {})
    socnav_cfg = None
    fixed_config_path = None
    if fixed_cfg.get("enabled", False):
        fixed_config_path = fixed_cfg.get("config_path")
        if not fixed_config_path:
            raise ValueError(
                "fixed_observation_space.config_path is required when the wrapper is enabled."
            )
        socnav_cfg = load_yaml(env_cfg["config_path"])
        if socnav_cfg.get("env", {}).get("get_padded_observations") is not True:
            raise ValueError("fixed_observation_space requires environment.config_path to set get_padded_observations: true.")

    import socnavgym  # noqa: F401 - registers SocNavGym-v1 with gym

    env = gym.make(env_cfg["id"], config=env_cfg["config_path"])

    # Project reward checks apply only when a real SocNavGym YAML is in use.
    config_path = Path(env_cfg["config_path"])
    if config_path.is_file():
        if socnav_cfg is None:
            socnav_cfg = load_yaml(config_path)
        _validate_project_static_warning_config(config, socnav_cfg)

    if wrappers.get("astar", {}).get("enabled", False):
        from global_planning.socnav_astar_wrapper import SocNavAStarWrapper

        env = SocNavAStarWrapper(env, config_path=wrappers["astar"].get("config_path"))

    if wrappers.get("nearest_wall_segments", {}).get("enabled", False):
        from navigation_features.nearest_wall_segment_wrapper import NearestWallSegmentWrapper

        wall_cfg = wrappers["nearest_wall_segments"]
        env = NearestWallSegmentWrapper(
            env,
            count=wall_cfg.get("count", 8),
            observation_key=wall_cfg.get("observation_key", "walls"),
            mode=wall_cfg.get("mode", "nearest"),
            include_boundary_walls=wall_cfg.get("include_boundary_walls", True),
        )

    if wrappers.get("navigation_features", {}).get("enabled", False):
        from navigation_features.coordinate_frame_waypoint_wrapper import CoordinateFrameWaypointWrapper

        env = CoordinateFrameWaypointWrapper(
            env,
            config_path=wrappers["navigation_features"].get("config_path"),
            config=wrappers["navigation_features"].get("config"),
        )

    if wrappers.get("warning_zone_visualization", {}).get("enabled", False):
        from custom_rewards.warning_zone_visualization_wrapper import WarningZoneVisualizationWrapper

        env = WarningZoneVisualizationWrapper(env, config_path=wrappers["warning_zone_visualization"].get("config_path"))

    if fixed_cfg.get("enabled", False):
        wall_cfg = wrappers.get("nearest_wall_segments", {})
        include_keys = fixed_cfg.get("include_keys", ("humans", "laptops", "tables", "plants", "walls"))
        wall_key = wall_cfg.get("observation_key", "walls")
        wall_wrapper_enabled = wall_cfg.get("enabled", False)
        # The wall wrapper already owns wall extraction, capacity, and padding.
        fixed_keys = tuple(key for key in include_keys if not (wall_wrapper_enabled and key == wall_key))
        env = FixedObservationSpaceWrapper(
            env,
            config_path=fixed_config_path,
            include_keys=fixed_keys,
            entity_feature_dim=config["architecture"].get("entity_feature_dim", 14),
            wall_count=0,
            wall_key=wall_key,
        )

    history_cfg = wrappers.get("observation_history", {})
    if history_cfg.get("enabled", False):
        architecture_cfg = config["architecture"]
        entity_keys = tuple(architecture_cfg.get("entity_keys", ("humans",)))
        default_temporal_keys = tuple(key for key in entity_keys if key == "humans")
        temporal_entity_keys = tuple(history_cfg.get("temporal_entity_keys", default_temporal_keys))
        wall_cfg = wrappers.get("nearest_wall_segments", {})
        wall_key = wall_cfg.get("observation_key", "walls")
        if (
            wall_cfg.get("enabled", False)
            and wall_key in temporal_entity_keys
            and wall_cfg.get("mode", "nearest") != "all"
        ):
            raise ValueError(
                "Genuine wall history requires nearest_wall_segments.mode: all; "
                "nearest-wall slots can change identity between steps."
            )
        env = ObservationHistoryWrapper(
            env,
            history_length=history_cfg["history_length"],
            entity_keys=entity_keys,
            entity_feature_dim=architecture_cfg.get("entity_feature_dim", 14),
            temporal_entity_keys=temporal_entity_keys,
        )

    if wrappers.get("diff_drive_action", {}).get("enabled", True):
        env = DifferentialDriveActionWrapper(env)

    if wrappers.get("drop_empty_observation_keys", {}).get("enabled", True):
        env = DropEmptyObservationKeysWrapper(env)

    if env_cfg.get("gymnasium_compatibility", True):
        env = GymnasiumCompatibilityWrapper(env)

    try:
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
    except AttributeError:
        pass
    return env


def validate_static_obstacle_reward_visibility(config, static_config):
    """Require every penalized static type to be visible to the policy."""
    if not static_config.enabled:
        return

    entity_keys = set(config.get("architecture", {}).get("entity_keys", ()))
    required_keys = {
        "tables": static_config.include_tables,
        "plants": static_config.include_plants,
        "laptops": static_config.include_laptops,
    }
    missing = [key for key, included in required_keys.items() if included and key not in entity_keys]

    wall_cfg = config.get("wrappers", {}).get("nearest_wall_segments", {})
    if static_config.include_walls:
        wall_key = wall_cfg.get("observation_key", "walls")
        if not wall_cfg.get("enabled", False) or wall_key not in entity_keys:
            missing.append(wall_key)
        elif static_config.include_boundary_walls and not wall_cfg.get("include_boundary_walls", True):
            raise ValueError(
                "Static warning zones include boundary walls, but nearest_wall_segments excludes them."
            )

    if missing:
        keys = ", ".join(sorted(set(missing)))
        raise ValueError(
            "Static warning zones require matching policy observations for: " + keys
        )


def _validate_project_static_warning_config(config, socnav_cfg):
    reward_file = socnav_cfg.get("env", {}).get("reward_file", "")
    if Path(str(reward_file)).name != "socnavgym_social_safety_reward.py":
        return

    from custom_rewards.static_obstacle_warning_zone import load_static_obstacle_warning_zone_config

    reward_config_path = Path(__file__).resolve().parents[1] / "custom_rewards" / "social_safety_reward_config.yaml"
    reward_values = load_yaml(reward_config_path).get("static_obstacle_warning_zone", {})
    validate_static_obstacle_reward_visibility(
        config,
        load_static_obstacle_warning_zone_config(reward_values),
    )


def make_vec_env(config: Dict[str, Any]):
    """Create a Stable-Baselines3 vectorized environment."""
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    env_cfg = config["environment"]
    num_envs = int(config["training"]["num_envs"])
    vec_env_type = config["training"].get("vec_env", "dummy")

    def make_one(rank: int):
        def _factory():
            env = make_socnav_env(config, rank=rank, eval_mode=False)
            if env_cfg.get("monitor", True):
                env = Monitor(env)
            return env

        return _factory

    factories = [make_one(rank) for rank in range(num_envs)]
    if vec_env_type == "subproc":
        return SubprocVecEnv(factories)
    if vec_env_type == "dummy":
        return DummyVecEnv(factories)
    raise ValueError('training.vec_env must be "dummy" or "subproc".')


def make_eval_env(config: Dict[str, Any]):
    """Create one monitored evaluation environment."""
    from stable_baselines3.common.monitor import Monitor

    env = make_socnav_env(config, rank=0, eval_mode=True)
    if config["environment"].get("monitor", True):
        env = Monitor(env)
    return env
