import gym
import numpy as np
from gym import spaces

from training_pipeline.utils import load_yaml

try:
    import gymnasium
except ImportError:
    gymnasium = None


_GymnasiumEnvBase = gymnasium.Env if gymnasium is not None else object


def derive_socnav_entity_counts(config_path, wall_count=0, wall_key="walls"):
    """Derive padded entity row counts using SocNavGym's observation-space formulas."""
    env_cfg = load_yaml(config_path)["env"]
    if env_cfg.get("get_padded_observations") is not True:
        raise ValueError("FixedObservationSpaceWrapper requires get_padded_observations: true in its target SocNavGym config.")

    h_l_interactions = (
        int(env_cfg["max_h_l_interactions"])
        + int(env_cfg["max_h_l_interactions_non_dispersing"])
    )
    h_h_dynamic = (
        int(env_cfg["max_h_h_dynamic_interactions"])
        + int(env_cfg["max_h_h_dynamic_interactions_non_dispersing"])
    )
    h_h_static = (
        int(env_cfg["max_h_h_static_interactions"])
        + int(env_cfg["max_h_h_static_interactions_non_dispersing"])
    )
    max_humans_per_h_h = int(env_cfg["max_human_in_h_h_interactions"])

    counts = {
        "humans": (
            int(env_cfg["max_static_humans"])
            + int(env_cfg["max_dynamic_humans"])
            + h_l_interactions
            + (h_h_dynamic * max_humans_per_h_h)
            + (h_h_static * max_humans_per_h_h)
        ),
        "laptops": int(env_cfg["max_laptops"]) + h_l_interactions,
        "tables": int(env_cfg["max_tables"]),
        "plants": int(env_cfg["max_plants"]),
    }
    counts[wall_key] = int(wall_count)
    return counts


def derive_socnav_entity_spaces(config_path, counts, entity_feature_dim=14, wall_key="walls"):
    """Build target entity spaces from SocNavGym's padded-observation bounds."""
    config = load_yaml(config_path)
    env_cfg = config["env"]
    try:
        map_x = float(env_cfg["max_map_x"])
        map_y = float(env_cfg["max_map_y"])
        max_advance_robot = float(env_cfg["max_advance_robot"])
        max_rotation = float(env_cfg["max_rotation"])
        timestep = float(config["episode"]["time_step"])
        human_diameter = float(config["human"]["human_diameter"])
        laptop_radius = _rectangle_radius(config["laptop"]["laptop_width"], config["laptop"]["laptop_length"])
        table_radius = _rectangle_radius(config["table"]["table_width"], config["table"]["table_length"])
        plant_radius = float(config["plant"]["plant_radius"])
    except KeyError as exc:
        raise ValueError(
            "FixedObservationSpaceWrapper target config is missing metadata needed "
            "to derive SocNavGym entity bounds."
        ) from exc

    max_xy = (map_x * np.sqrt(2), map_y * np.sqrt(2))
    max_advance_human = float(env_cfg["max_advance_human"])
    row_bounds = {
        "humans": (
            [0, 0, 0, 0, 0, 0, -max_xy[0], -max_xy[1], -1, -1, -human_diameter / 2, -(max_advance_human + max_advance_robot) * np.sqrt(2), -2 * np.pi / timestep, 0],
            [1, 1, 1, 1, 1, 1, max_xy[0], max_xy[1], 1, 1, human_diameter / 2, (max_advance_human + max_advance_robot) * np.sqrt(2), 2 * np.pi / timestep, 1],
        ),
        "laptops": _static_entity_row_bounds(max_xy, laptop_radius, max_advance_robot, max_rotation),
        "tables": _static_entity_row_bounds(max_xy, table_radius, max_advance_robot, max_rotation),
        "plants": _static_entity_row_bounds(max_xy, plant_radius, max_advance_robot, max_rotation),
    }

    target_spaces = {}
    for key, count in counts.items():
        if key == wall_key or key not in row_bounds:
            continue
        low_row, high_row = row_bounds[key]
        target_spaces[key] = spaces.Box(
            low=np.tile(np.asarray(low_row, dtype=np.float32), int(count)),
            high=np.tile(np.asarray(high_row, dtype=np.float32), int(count)),
            shape=(int(count) * int(entity_feature_dim),),
            dtype=np.float32,
        )
    return target_spaces


def _rectangle_radius(width, length):
    return float(np.hypot(float(width), float(length)) / 2)


def _static_entity_row_bounds(max_xy, radius, max_advance_robot, max_rotation):
    low = [0, 0, 0, 0, 0, 0, -max_xy[0], -max_xy[1], -1, -1, -radius, -max_advance_robot * np.sqrt(2), -max_rotation, 0]
    high = [1, 1, 1, 1, 1, 1, max_xy[0], max_xy[1], 1, 1, radius, max_advance_robot * np.sqrt(2), max_rotation, 1]
    return low, high


class FixedObservationSpaceWrapper(gym.Wrapper):
    """Pad selected entity observations to counts derived from a target SocNavGym config."""

    def __init__(
        self,
        env,
        config_path,
        include_keys,
        entity_feature_dim=14,
        wall_count=0,
        wall_key="walls",
    ):
        super().__init__(env)
        self.entity_feature_dim = int(entity_feature_dim)
        if self.entity_feature_dim != 14:
            raise ValueError("SocNavGym fixed entity bounds require entity_feature_dim: 14.")

        derived_counts = derive_socnav_entity_counts(config_path, wall_count=wall_count, wall_key=wall_key)
        unknown_keys = [key for key in include_keys if key not in derived_counts]
        if unknown_keys:
            raise ValueError(f"Unknown fixed observation keys: {unknown_keys}")
        self.entity_counts = {
            key: int(derived_counts[key])
            for key in include_keys
        }
        self.target_entity_spaces = derive_socnav_entity_spaces(
            config_path,
            self.entity_counts,
            entity_feature_dim=self.entity_feature_dim,
            wall_key=wall_key,
        )
        self.observation_space = self._build_observation_space()

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        if isinstance(result, tuple) and len(result) == 2:
            obs, info = result
            return self._fix_observation(obs), info
        return self._fix_observation(result)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if isinstance(info, dict) and "terminal_observation" in info:
            info = dict(info)
            info["terminal_observation"] = self._fix_observation(info["terminal_observation"])
        return self._fix_observation(obs), reward, terminated, truncated, info

    def _build_observation_space(self):
        if not isinstance(self.env.observation_space, spaces.Dict):
            return self.env.observation_space

        obs_spaces = dict(self.env.observation_space.spaces)
        for key, count in self.entity_counts.items():
            obs_spaces[key] = self._fixed_entity_space(key, obs_spaces.get(key), count)
        return spaces.Dict(obs_spaces)

    def _fixed_entity_space(self, key, source_space, count):
        size = count * self.entity_feature_dim
        if count == 0:
            return spaces.Box(
                low=np.zeros(0, dtype=np.float32),
                high=np.zeros(0, dtype=np.float32),
                shape=(0,),
                dtype=np.float32,
            )

        target_space = self.target_entity_spaces.get(key)
        if target_space is not None:
            return target_space

        if isinstance(source_space, spaces.Box):
            low = np.asarray(source_space.low, dtype=np.float32).reshape(-1)
            high = np.asarray(source_space.high, dtype=np.float32).reshape(-1)
            if low.size >= self.entity_feature_dim and low.size % self.entity_feature_dim == 0:
                # SocNavGym repeats the same per-entity bounds for padded entity rows.
                return spaces.Box(
                    low=np.tile(low[: self.entity_feature_dim], count),
                    high=np.tile(high[: self.entity_feature_dim], count),
                    shape=(size,),
                    dtype=np.float32,
                )

        return spaces.Box(
            low=np.full(size, -np.inf, dtype=np.float32),
            high=np.full(size, np.inf, dtype=np.float32),
            shape=(size,),
            dtype=np.float32,
        )

    def _fix_observation(self, obs):
        if not isinstance(obs, dict):
            return obs

        fixed = dict(obs)
        for key, count in self.entity_counts.items():
            fixed[key] = self._pad_entity_rows(key, fixed.get(key), count)
        return fixed

    def _pad_entity_rows(self, key, values, count):
        if values is None:
            rows = np.zeros((0, self.entity_feature_dim), dtype=np.float32)
        else:
            flat = np.asarray(values, dtype=np.float32).reshape(-1)
            if flat.size % self.entity_feature_dim != 0:
                raise ValueError(f"{key} observation size is not divisible by entity feature dim.")
            rows = flat.reshape(-1, self.entity_feature_dim)

        if rows.shape[0] > count:
            raise ValueError(f"{key} has {rows.shape[0]} rows, but fixed observation capacity is {count}.")

        padded = np.zeros((count, self.entity_feature_dim), dtype=np.float32)
        padded[: rows.shape[0]] = rows
        return padded.reshape(-1)


class DifferentialDriveActionWrapper(gym.ActionWrapper):
    """
    Expose a 2-D differential-drive action to PPO.

    SocNavGym expects a 3-D normalized action:

        [linear_velocity, lateral_velocity, angular_velocity]

    For a differential-drive robot, lateral velocity must be zero. This wrapper
    lets PPO output only:

        [linear_velocity, angular_velocity]

    and inserts the zero lateral component before calling the base env.
    """

    def __init__(self, env):
        super().__init__(env)
        # socnavgym action space lie in the range of [-1, 1].
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

    def action(self, action):
        action = np.asarray(action, dtype=np.float32)
        return np.array([action[0], 0.0, action[1]], dtype=np.float32)


class DropEmptyObservationKeysWrapper(gym.Wrapper):
    """Remove zero-length dict observation keys before Stable-Baselines3 sees them."""

    def __init__(self, env):
        super().__init__(env)
        if not isinstance(env.observation_space, spaces.Dict):
            self.keep_keys = None
            return

        self.keep_keys = [
            key for key, space in env.observation_space.spaces.items()
            if int(np.prod(space.shape)) > 0
        ]
        self.observation_space = spaces.Dict({key: env.observation_space.spaces[key] for key in self.keep_keys})

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        if isinstance(result, tuple) and len(result) == 2:
            obs, info = result
            return self._filter_observation(obs), info
        return self._filter_observation(result)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if isinstance(info, dict) and "terminal_observation" in info:
            info = dict(info)
            info["terminal_observation"] = self._filter_observation(info["terminal_observation"])
        return self._filter_observation(obs), reward, terminated, truncated, info

    def _filter_observation(self, obs):
        if self.keep_keys is None or not isinstance(obs, dict):
            return obs
        return {key: obs[key] for key in self.keep_keys if key in obs}


class GymnasiumCompatibilityWrapper(_GymnasiumEnvBase):
    """
    Convert a Gym-style SocNavGym environment to a Gymnasium-style environment.

    Stable-Baselines3 v2 uses Gymnasium internally. This adapter keeps the
    training pipeline independent of shimmy for the common Box/Dict spaces used
    by SocNavGym.
    """

    metadata = {}

    def __init__(self, env):
        if gymnasium is None:
            exc = ImportError("No module named 'gymnasium'")
            raise ImportError("gymnasium is required when environment.gymnasium_compatibility is true.") from exc

        self.env = env
        self.action_space = self._convert_space(env.action_space)
        self.observation_space = self._convert_space(env.observation_space)
        self.metadata = getattr(env, "metadata", {})
        self.render_mode = getattr(env, "render_mode", None)

    def reset(self, *, seed=None, options=None):
        kwargs = {}
        if seed is not None:
            kwargs["seed"] = seed
        result = self.env.reset(**kwargs)
        if isinstance(result, tuple) and len(result) == 2:
            return result
        return result, {}

    def step(self, action):
        return self.env.step(action)

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def __getattr__(self, name):
        return getattr(self.env, name)

    def _convert_space(self, space):
        gymnasium_spaces = gymnasium.spaces
        if isinstance(space, spaces.Box):
            return gymnasium_spaces.Box(low=space.low, high=space.high, shape=space.shape, dtype=space.dtype)
        if isinstance(space, spaces.Dict):
            return gymnasium_spaces.Dict({key: self._convert_space(value) for key, value in space.spaces.items()})
        if isinstance(space, spaces.Discrete):
            return gymnasium_spaces.Discrete(space.n)
        raise TypeError(f"Unsupported space type for Gymnasium compatibility: {type(space).__name__}")
