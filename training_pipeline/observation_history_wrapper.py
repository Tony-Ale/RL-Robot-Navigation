from collections import deque

import gym
import numpy as np
from gym import spaces


class ObservationHistoryWrapper(gym.Wrapper):
    """
    Expose fixed observation histories for temporal feature extractors.

    Robot and waypoint observations retain their real recent history. Entity
    keys listed in ``temporal_entity_keys`` do the same; other entity keys
    repeat their current rows across time for slot-unstable observations.
    """

    FRAME_KEYS = ("robot", "waypoint_features")

    def __init__(
        self,
        env,
        history_length,
        entity_keys,
        entity_feature_dim=14,
        temporal_entity_keys=None,
    ):
        super().__init__(env)
        self.history_length = int(history_length)
        self.entity_keys = tuple(entity_keys)
        self.temporal_entity_keys = frozenset(
            (key for key in self.entity_keys if key == "humans")
            if temporal_entity_keys is None
            else temporal_entity_keys
        )
        self.entity_feature_dim = int(entity_feature_dim)
        if self.history_length <= 0:
            raise ValueError("observation_history.history_length must be greater than zero.")
        if not isinstance(env.observation_space, spaces.Dict):
            raise TypeError("ObservationHistoryWrapper requires a Dict observation space.")
        unknown_temporal_keys = self.temporal_entity_keys.difference(self.entity_keys)
        if unknown_temporal_keys:
            raise ValueError(
                "observation_history.temporal_entity_keys must be included in architecture.entity_keys: "
                f"{sorted(unknown_temporal_keys)}"
            )

        self._histories = {}
        self.observation_space = self._build_observation_space()

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        if isinstance(result, tuple) and len(result) == 2:
            obs, info = result
            return self._process_observation(obs, reset=True), info
        return self._process_observation(result, reset=True)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._process_observation(obs), reward, terminated, truncated, info

    def _build_observation_space(self):
        observation_spaces = dict(self.env.observation_space.spaces)
        for key in self.FRAME_KEYS:
            if key in observation_spaces:
                observation_spaces[key] = self._frame_history_space(key, observation_spaces[key])
        for key in self.entity_keys:
            if key in observation_spaces:
                observation_spaces[key] = self._entity_history_space(key, observation_spaces[key])
        return spaces.Dict(observation_spaces)

    def _frame_history_space(self, key, source_space):
        if not isinstance(source_space, spaces.Box) or len(source_space.shape) != 1:
            raise ValueError(f"Observation key '{key}' must use a one-dimensional Box space before history is added.")
        low = np.repeat(source_space.low[None, :], self.history_length, axis=0)
        high = np.repeat(source_space.high[None, :], self.history_length, axis=0)
        return spaces.Box(low=low, high=high, dtype=source_space.dtype)

    def _entity_history_space(self, key, source_space):
        if not isinstance(source_space, spaces.Box):
            raise ValueError(f"Entity observation key '{key}' must use a Box space.")
        low_rows = self._entity_rows(key, source_space.low)
        high_rows = self._entity_rows(key, source_space.high)
        low = np.repeat(low_rows[:, None, :], self.history_length, axis=1)
        high = np.repeat(high_rows[:, None, :], self.history_length, axis=1)
        return spaces.Box(low=low, high=high, dtype=source_space.dtype)

    def _process_observation(self, obs, reset=False):
        if not isinstance(obs, dict):
            return obs
        if reset:
            self._histories.clear()

        processed = dict(obs)
        for key in self.FRAME_KEYS:
            if key in processed:
                processed[key] = self._frame_history(key, processed[key], reset)

        for key in self.entity_keys:
            if key not in processed:
                continue
            rows = self._entity_rows(key, processed[key])
            if key in self.temporal_entity_keys:
                processed[key] = self._dynamic_entity_history(key, rows, reset)
            else:
                # Slot-unstable observations must not create false entity motion.
                processed[key] = np.repeat(rows[:, None, :], self.history_length, axis=1)
        return processed

    def _frame_history(self, key, values, reset):
        frame = np.asarray(values, dtype=np.float32).reshape(-1)
        history = self._history_for(key, frame, reset)
        return np.stack(history, axis=0)

    def _dynamic_entity_history(self, key, rows, reset):
        history = self._history_for(key, rows, reset)
        return np.stack(history, axis=1)

    def _history_for(self, key, values, reset):
        if reset or key not in self._histories:
            self._histories[key] = deque(
                (np.array(values, copy=True) for _ in range(self.history_length)),
                maxlen=self.history_length,
            )
        else:
            self._histories[key].append(np.array(values, copy=True))
        return self._histories[key]

    def _entity_rows(self, key, values):
        flat = np.asarray(values, dtype=np.float32).reshape(-1)
        if flat.size % self.entity_feature_dim != 0:
            raise ValueError(f"{key} observation size is not divisible by entity feature dim.")
        return flat.reshape(-1, self.entity_feature_dim)
