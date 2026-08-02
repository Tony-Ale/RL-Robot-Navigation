import tempfile
import unittest
from pathlib import Path

import gym
import numpy as np
import socnavgym  # noqa: F401 - registers SocNavGym-v1 with gym
import yaml

from environment_inspection.wall_segment_analysis import (
    calculate_wall_segment_capacity,
    count_live_wall_segments,
)
from navigation_features.nearest_wall_segment_wrapper import NearestWallSegmentWrapper
from navigation_features.wall_geometry import is_boundary_wall
from training_pipeline.utils import load_yaml


ROOT = Path(__file__).resolve().parents[1]
TARGET_CONFIG = ROOT / "env_configs" / "env_main.yaml"


def _wall_only_config():
    config = load_yaml(str(TARGET_CONFIG))
    for entity in ("static_humans", "dynamic_humans", "tables", "plants", "laptops"):
        config["env"][f"min_{entity}"] = 0
        config["env"][f"max_{entity}"] = 0
    return config


class TestWallSegmentAnalysis(unittest.TestCase):
    def test_current_target_capacity_is_24(self):
        """Testing: the configured 10 m corridor room has at most 24 wall rows."""
        config = load_yaml(str(TARGET_CONFIG))

        self.assertEqual(calculate_wall_segment_capacity(config, 3.0), 24)
        self.assertEqual(calculate_wall_segment_capacity(config, 5.0), 14)
        self.assertEqual(calculate_wall_segment_capacity(config, 15.0), 8)

    def test_corridor_only_capacity_excludes_boundary_segments(self):
        """Testing: corridor-only capacity follows segment size without hardcoded padding."""
        config = load_yaml(str(TARGET_CONFIG))

        self.assertEqual(calculate_wall_segment_capacity(config, 3.0, False), 8)
        self.assertEqual(calculate_wall_segment_capacity(config, 5.0, False), 6)
        self.assertEqual(calculate_wall_segment_capacity(config, 15.0, False), 4)

    def test_live_wall_rows_reach_but_do_not_exceed_capacity(self):
        """Testing: fixed live seeds agree with the calculated wall capacity."""
        config = _wall_only_config()
        capacity = calculate_wall_segment_capacity(config, 3.0)
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "wall_capacity.yaml"
            config_path.write_text(yaml.safe_dump(config, sort_keys=False))
            env = gym.make("SocNavGym-v1", config=str(config_path))
            try:
                observed = []
                observed_corridors = []
                for seed in range(10):
                    env.reset(seed=seed)
                    observed.append(count_live_wall_segments(env))
                    observed_corridors.append(count_live_wall_segments(env, include_boundary_walls=False))
            finally:
                env.close()

        self.assertEqual(max(observed), capacity)
        self.assertTrue(all(value <= capacity for value in observed))
        corridor_capacity = calculate_wall_segment_capacity(config, 3.0, False)
        self.assertEqual(max(observed_corridors), corridor_capacity)
        self.assertTrue(all(value <= corridor_capacity for value in observed_corridors))

    def test_all_mode_matches_live_socnavgym_order_across_steps(self):
        """Testing: all-mode rows follow stable SocNavGym wall order within an episode."""
        config = _wall_only_config()
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "wall_order.yaml"
            config_path.write_text(yaml.safe_dump(config, sort_keys=False))
            env = NearestWallSegmentWrapper(
                gym.make("SocNavGym-v1", config=str(config_path)),
                count=8,
                mode="all",
                include_boundary_walls=False,
            )
            try:
                observation, _ = env.reset(seed=0)
                base_env = env.unwrapped
                wall_identity = tuple(id(wall) for wall in base_env.walls)
                expected = _direct_wall_rows(base_env, include_boundary_walls=False)
                np.testing.assert_allclose(observation["walls"][: expected.size], expected)
                initial_rows = observation["walls"].copy()
                initial_geometry = _wall_geometry(base_env)

                observation, *_ = env.step(np.array([0.4, 0.0, 0.2], dtype=np.float32))
                self.assertEqual(tuple(id(wall) for wall in base_env.walls), wall_identity)
                expected = _direct_wall_rows(base_env, include_boundary_walls=False)
                np.testing.assert_allclose(observation["walls"][: expected.size], expected)

                repeated, _ = env.reset(seed=0)
                self.assertEqual(_wall_geometry(base_env), initial_geometry)
                np.testing.assert_allclose(repeated["walls"], initial_rows)
            finally:
                env.close()


def _direct_wall_rows(base_env, include_boundary_walls=True):
    rows = [
        np.asarray(base_env._get_entity_obs(wall), dtype=np.float32)
        for wall in base_env.walls
        if include_boundary_walls or not is_boundary_wall(wall, base_env.MAP_X, base_env.MAP_Y)
    ]
    return np.concatenate(rows) if rows else np.zeros(0, dtype=np.float32)


def _wall_geometry(base_env):
    return tuple(
        (float(wall.x), float(wall.y), float(wall.orientation), float(wall.length))
        for wall in base_env.walls
    )


if __name__ == "__main__":
    unittest.main()
