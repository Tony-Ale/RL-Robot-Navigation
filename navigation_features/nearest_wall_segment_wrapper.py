import numpy as np
import gym
from gym import spaces

from navigation_features.wall_geometry import is_boundary_wall


class NearestWallSegmentWrapper(gym.Wrapper):
    """Expose a fixed number of nearest or all SocNavGym wall segments."""

    ENTITY_FEATURE_DIM = 14

    MODES = frozenset({"nearest", "all"})

    def __init__(
        self,
        env,
        count=8,
        observation_key="walls",
        mode="nearest",
        include_boundary_walls=True,
    ):
        super().__init__(env)
        self.count = int(count)
        self.observation_key = observation_key
        self.mode = str(mode).lower()
        self.include_boundary_walls = bool(include_boundary_walls)
        if self.count <= 0:
            raise ValueError("wall segment count must be greater than zero.")
        if self.mode not in self.MODES:
            raise ValueError('wall segment mode must be "nearest" or "all".')
        self.observation_space = self._build_observation_space()

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._process_observation(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._process_observation(obs), reward, terminated, truncated, info

    def _build_observation_space(self):
        if not hasattr(self.env.observation_space, "spaces"):
            return self.env.observation_space

        obs_spaces = dict(self.env.observation_space.spaces)
        size = self.count * self.ENTITY_FEATURE_DIM
        obs_spaces[self.observation_key] = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(size,),
            dtype=np.float32,
        )
        return spaces.Dict(obs_spaces)

    def _process_observation(self, obs):
        if not isinstance(obs, dict):
            return obs

        processed = dict(obs)
        processed[self.observation_key] = self._wall_segments().reshape(-1)
        return processed

    def _wall_segments(self):
        rows, clearances = self._wall_segment_candidates(include_clearances=self.mode == "nearest")
        padded = np.zeros((self.count, self.ENTITY_FEATURE_DIM), dtype=np.float32)
        if rows.size == 0:
            return padded

        if self.mode == "all":
            if rows.shape[0] > self.count:
                raise ValueError(
                    f"walls produced {rows.shape[0]} segments, but all-mode capacity is {self.count}."
                )
            selected = rows
        else:
            selected = rows[np.argsort(clearances, kind="stable")[: self.count]]

        padded[: selected.shape[0]] = selected
        return padded

    def _wall_segment_candidates(self, include_clearances=True):
        base_env = self.unwrapped
        get_entity_obs = getattr(base_env, "_get_entity_obs", None)
        walls = self._selected_walls(base_env)
        if get_entity_obs is None or not walls:
            return self._empty_candidates()

        rows = []
        clearances = []
        robot_radius = float(getattr(getattr(base_env, "robot", None), "radius", 0.0))
        for wall in walls:
            wall_obs = np.asarray(get_entity_obs(wall), dtype=np.float32)
            if wall_obs.size == 0:
                continue
            if wall_obs.size % self.ENTITY_FEATURE_DIM != 0:
                raise ValueError("SocNavGym wall observation size is not divisible by entity feature dim.")
            wall_rows = wall_obs.reshape(-1, self.ENTITY_FEATURE_DIM)
            rows.append(wall_rows)
            if include_clearances:
                clearances.append(self._surface_clearances(wall_rows, float(wall.thickness), robot_radius))

        if not rows:
            return self._empty_candidates()
        all_rows = np.concatenate(rows, axis=0)
        all_clearances = np.concatenate(clearances) if clearances else np.zeros(0, dtype=np.float32)
        return all_rows, all_clearances

    def _selected_walls(self, base_env):
        walls = getattr(base_env, "walls", [])
        if self.include_boundary_walls or not walls:
            return walls

        shape = getattr(base_env, "shape", None)
        if shape not in ("square", "rectangle"):
            raise ValueError("Boundary-wall filtering supports square and rectangle rooms.")
        map_x = float(base_env.MAP_X)
        map_y = float(base_env.MAP_Y)
        return [wall for wall in walls if not is_boundary_wall(wall, map_x, map_y)]

    @staticmethod
    def _surface_clearances(rows, wall_thickness, robot_radius):
        """Return robot-to-rectangle clearance for each finite wall segment."""
        x = rows[:, 6]
        y = rows[:, 7]
        sin_theta = rows[:, 8]
        cos_theta = rows[:, 9]
        half_length = np.abs(rows[:, 10])

        # Project the robot-relative segment centre onto the segment's local axes.
        along = np.abs(x * cos_theta + y * sin_theta)
        across = np.abs(-x * sin_theta + y * cos_theta)
        outside_length = np.maximum(along - half_length, 0.0)
        outside_width = np.maximum(across - wall_thickness / 2.0, 0.0)
        return np.hypot(outside_length, outside_width) - robot_radius

    @classmethod
    def _empty_candidates(cls):
        return np.zeros((0, cls.ENTITY_FEATURE_DIM), dtype=np.float32), np.zeros(0, dtype=np.float32)
