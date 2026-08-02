import json
import io
from pathlib import Path
import tempfile
import textwrap
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import gym
import numpy as np
import socnavgym  # noqa: F401 - registers SocNavGym-v1 with gym
import torch
import yaml
from gym import spaces

from training_pipeline.action_wrappers import (
    DifferentialDriveActionWrapper,
    DropEmptyObservationKeysWrapper,
    FixedObservationSpaceWrapper,
    GymnasiumCompatibilityWrapper,
    derive_socnav_entity_counts,
)
from training_pipeline.architecture_extractor import (
    ArchitectureFeaturesExtractor,
    architecture_feature_dim,
    effective_robot_input_dim,
)
from training_pipeline.callbacks import (
    CSVMetricWriter,
    NavigationEvaluationCallback,
    NavigationTrainingCallback,
    TrainingRenderCallback,
    record_training_time_session,
)
from training_pipeline.env_factory import make_socnav_env
from training_pipeline.observation_history_wrapper import ObservationHistoryWrapper
from training_pipeline.train import build_callbacks, validate_architecture_entity_keys
from training_pipeline.utils import (
    REWARD_CONFIG_SNAPSHOT_NAME,
    load_yaml,
    make_run_dir,
    record_training_seed_session,
)


ROOT = Path(__file__).resolve().parents[1]


def write_fixed_observation_target_config(path, env_overrides=None):
    """Write a complete SocNavGym-like target config for fixed-observation tests."""
    config = {
        "episode": {"time_step": 0.25},
        "human": {"human_diameter": 0.72},
        "laptop": {"laptop_width": 0.4, "laptop_length": 0.6},
        "plant": {"plant_radius": 0.4},
        "table": {"table_width": 0.75, "table_length": 1.5},
        "env": {
            "get_padded_observations": True,
            "max_advance_human": 1.0,
            "max_advance_robot": 1.0,
            "max_rotation": 1.2,
            "max_map_x": 10,
            "max_map_y": 10,
            "max_static_humans": 0,
            "max_dynamic_humans": 0,
            "max_tables": 0,
            "max_plants": 0,
            "max_laptops": 0,
            "max_h_h_dynamic_interactions": 0,
            "max_h_h_dynamic_interactions_non_dispersing": 0,
            "max_h_h_static_interactions": 0,
            "max_h_h_static_interactions_non_dispersing": 0,
            "max_human_in_h_h_interactions": 0,
            "max_h_l_interactions": 0,
            "max_h_l_interactions_non_dispersing": 0,
        },
    }
    if env_overrides:
        config["env"].update(env_overrides)
    path.write_text(yaml.safe_dump(config, sort_keys=False))


class DummySocNavActionEnv(gym.Env):
    """Minimal env used to test the differential-drive action wrapper."""

    def __init__(self):
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    def reset(self, **kwargs):
        return np.zeros(1, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(1, dtype=np.float32), 0.0, False, False, {"raw_action": action}


class DummyDictObservationEnv(gym.Env):
    """Minimal dict-observation env with empty keys."""

    def __init__(self):
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Dict(
            {
                "robot": spaces.Box(low=-1.0, high=1.0, shape=(9,), dtype=np.float32),
                "humans": spaces.Box(low=-1.0, high=1.0, shape=(14,), dtype=np.float32),
                "tables": spaces.Box(low=-1.0, high=1.0, shape=(0,), dtype=np.float32),
            }
        )

    def reset(self, **kwargs):
        return self._obs(), {}

    def step(self, action):
        info = {"terminal_observation": self._obs()}
        return self._obs(), 0.0, False, True, info

    def _obs(self):
        return {
            "robot": np.zeros(9, dtype=np.float32),
            "humans": np.zeros(14, dtype=np.float32),
            "tables": np.zeros(0, dtype=np.float32),
        }


class DummyFixedObservationEnv(gym.Env):
    """Minimal dict-observation env used to test fixed entity padding."""

    def __init__(self, humans_rows=2, bounded_humans=False):
        self.humans_rows = humans_rows
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        human_low = np.full(humans_rows * 14, -np.inf, dtype=np.float32)
        human_high = np.full(humans_rows * 14, np.inf, dtype=np.float32)
        if bounded_humans:
            row_low = np.arange(14, dtype=np.float32) * -1.0
            row_high = np.arange(14, dtype=np.float32) + 1.0
            human_low = np.tile(row_low, humans_rows)
            human_high = np.tile(row_high, humans_rows)
        self.observation_space = spaces.Dict(
            {
                "robot": spaces.Box(low=-1.0, high=1.0, shape=(9,), dtype=np.float32),
                "humans": spaces.Box(low=human_low, high=human_high, shape=(humans_rows * 14,), dtype=np.float32),
            }
        )

    def reset(self, **kwargs):
        return self._obs(), {}

    def step(self, action):
        info = {"terminal_observation": self._obs()}
        return self._obs(), 0.0, False, True, info

    def _obs(self):
        return {
            "robot": np.zeros(9, dtype=np.float32),
            "humans": np.arange(self.humans_rows * 14, dtype=np.float32),
        }


class DummyHistoryObservationEnv(gym.Env):
    """Dict env with dynamic humans and static tables for history tests."""

    def __init__(self):
        self.step_number = 0
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Dict(
            {
                "robot": spaces.Box(low=-10.0, high=10.0, shape=(9,), dtype=np.float32),
                "waypoint_features": spaces.Box(low=-10.0, high=10.0, shape=(8,), dtype=np.float32),
                "humans": spaces.Box(low=-10.0, high=10.0, shape=(28,), dtype=np.float32),
                "tables": spaces.Box(low=-10.0, high=10.0, shape=(14,), dtype=np.float32),
            }
        )

    def reset(self, **kwargs):
        self.step_number = 0
        return self._obs(), {}

    def step(self, action):
        self.step_number += 1
        return self._obs(), 0.0, False, False, {}

    def _obs(self):
        robot = np.zeros(9, dtype=np.float32)
        robot[0] = self.step_number
        waypoints = np.full(8, self.step_number, dtype=np.float32)
        humans = np.zeros((2, 14), dtype=np.float32)
        humans[0, 0] = 1.0
        humans[0, 6] = self.step_number
        tables = np.zeros((1, 14), dtype=np.float32)
        tables[0, 3] = 1.0
        tables[0, 6] = self.step_number
        return {
            "robot": robot,
            "waypoint_features": waypoints,
            "humans": humans.reshape(-1),
            "tables": tables.reshape(-1),
        }


class DummyModel:
    """Minimal model stub for callback integration tests."""

    def __init__(self):
        self.saved_paths = []
        self.logger = DummyLogger()

    def save(self, path):
        self.saved_paths.append(Path(path))


class DummyLogger:
    """Minimal logger stub for callback integration tests."""

    def __init__(self):
        self.records = {}

    def record(self, key, value):
        self.records[key] = value


class TestTrainingPipeline(unittest.TestCase):
    """Tests for reusable training-pipeline utilities."""

    def test_fixed_observation_target_config_is_required_only_when_enabled(self):
        """Testing: only an enabled fixed wrapper requires an explicit target config."""
        print("Testing: fixed observation target config follows the wrapper enabled flag")
        config = {
            "experiment": {"seed": 1},
            "environment": {
                "id": "DummyEnv-v0",
                "config_path": "unused.yaml",
                "use_socnavgym_clone": False,
                "gymnasium_compatibility": False,
            },
            "architecture": {"entity_feature_dim": 14},
            "wrappers": {
                "fixed_observation_space": {"enabled": True, "config_path": None},
                "diff_drive_action": {"enabled": False},
                "drop_empty_observation_keys": {"enabled": False},
            },
        }

        with self.assertRaisesRegex(ValueError, "config_path is required"):
            make_socnav_env(config)

        config["wrappers"]["fixed_observation_space"]["enabled"] = False
        with patch("training_pipeline.env_factory.gym.make", return_value=DummySocNavActionEnv()):
            env = make_socnav_env(config)
        self.assertIsInstance(env, DummySocNavActionEnv)

    def test_nearest_wall_mode_rejects_genuine_wall_history(self):
        """Testing: distance-ranked wall slots cannot be treated as stable history."""
        config = {
            "experiment": {"seed": 1},
            "environment": {
                "id": "DummyEnv-v0",
                "config_path": "unused.yaml",
                "use_socnavgym_clone": False,
                "gymnasium_compatibility": False,
            },
            "architecture": {"entity_keys": ["walls"], "entity_feature_dim": 14},
            "wrappers": {
                "nearest_wall_segments": {
                    "enabled": True,
                    "mode": "nearest",
                    "count": 4,
                    "observation_key": "walls",
                },
                "observation_history": {
                    "enabled": True,
                    "history_length": 3,
                    "temporal_entity_keys": ["walls"],
                },
                "fixed_observation_space": {"enabled": False},
                "diff_drive_action": {"enabled": False},
                "drop_empty_observation_keys": {"enabled": False},
            },
        }

        with patch("training_pipeline.env_factory.gym.make", return_value=DummySocNavActionEnv()):
            with self.assertRaisesRegex(ValueError, "mode: all"):
                make_socnav_env(config)

    def test_diff_drive_action_wrapper_inserts_zero_lateral_velocity(self):
        """Testing: PPO 2-D action becomes SocNavGym [linear, 0, angular]."""
        print("Testing: differential-drive action wrapper maps 2-D PPO action to 3-D SocNavGym action")
        env = DifferentialDriveActionWrapper(DummySocNavActionEnv())

        mapped = env.action(np.array([0.25, -0.75], dtype=np.float32))

        np.testing.assert_allclose(mapped, np.array([0.25, 0.0, -0.75], dtype=np.float32))
        self.assertEqual(env.action_space.shape, (2,))

    def test_gymnasium_compatibility_wrapper_converts_spaces_and_preserves_step_api(self):
        """Testing: Gym env is adapted to Gymnasium spaces for Stable-Baselines3."""
        print("Testing: Gymnasium compatibility wrapper converts action/observation spaces")
        env = GymnasiumCompatibilityWrapper(DummySocNavActionEnv())

        obs, info = env.reset(seed=123)
        step = env.step(np.array([0.1, 0.0, -0.2], dtype=np.float32))

        self.assertEqual(env.action_space.shape, (3,))
        self.assertEqual(env.observation_space.shape, (1,))
        self.assertEqual(obs.shape, (1,))
        self.assertEqual(info, {})
        self.assertEqual(len(step), 5)

    def test_drop_empty_observation_keys_removes_zero_length_spaces_and_observations(self):
        """Testing: zero-length observation keys are removed before SB3 sees them."""
        print("Testing: empty observation keys are dropped from spaces, observations, and terminal observations")
        env = DropEmptyObservationKeysWrapper(DummyDictObservationEnv())

        obs, _ = env.reset()
        step_obs, _, _, _, info = env.step(np.zeros(1, dtype=np.float32))

        self.assertEqual(set(env.observation_space.spaces), {"robot", "humans"})
        self.assertEqual(set(obs), {"robot", "humans"})
        self.assertEqual(set(step_obs), {"robot", "humans"})
        self.assertEqual(set(info["terminal_observation"]), {"robot", "humans"})

    def test_observation_history_tracks_all_slot_stable_entities(self):
        """Testing: genuine histories slide for every configured stable entity key."""
        print("Testing: observation history tracks humans and static entities")
        env = ObservationHistoryWrapper(
            DummyHistoryObservationEnv(),
            history_length=3,
            entity_keys=("humans", "tables"),
            temporal_entity_keys=("humans", "tables"),
        )

        initial, _ = env.reset()
        step_one, _, _, _, _ = env.step(np.zeros(1, dtype=np.float32))
        step_two, _, _, _, _ = env.step(np.zeros(1, dtype=np.float32))

        self.assertEqual(env.observation_space.spaces["robot"].shape, (3, 9))
        self.assertEqual(env.observation_space.spaces["humans"].shape, (2, 3, 14))
        self.assertEqual(env.observation_space.spaces["tables"].shape, (1, 3, 14))
        np.testing.assert_array_equal(initial["robot"][:, 0], [0.0, 0.0, 0.0])
        np.testing.assert_array_equal(step_one["robot"][:, 0], [0.0, 0.0, 1.0])
        np.testing.assert_array_equal(step_two["humans"][0, :, 6], [0.0, 1.0, 2.0])
        np.testing.assert_array_equal(step_two["humans"][1], np.zeros((3, 14), dtype=np.float32))
        np.testing.assert_array_equal(step_two["tables"][0, :, 6], [0.0, 1.0, 2.0])

    def test_observation_history_repeats_slot_unstable_entities(self):
        """Testing: excluded entity keys repeat current rows instead of false history."""
        env = ObservationHistoryWrapper(
            DummyHistoryObservationEnv(),
            history_length=3,
            entity_keys=("humans", "tables"),
            temporal_entity_keys=("humans",),
        )

        env.reset()
        env.step(np.zeros(1, dtype=np.float32))
        observation, _, _, _, _ = env.step(np.zeros(1, dtype=np.float32))

        np.testing.assert_array_equal(observation["tables"][0, :, 6], [2.0, 2.0, 2.0])

    def test_observation_history_rejects_unknown_temporal_keys(self):
        """Testing: temporal keys must also be architecture entity keys."""
        with self.assertRaisesRegex(ValueError, "must be included"):
            ObservationHistoryWrapper(
                DummyHistoryObservationEnv(),
                history_length=3,
                entity_keys=("humans",),
                temporal_entity_keys=("tables",),
            )

    def test_observation_history_reset_clears_previous_episode(self):
        """Testing: history buffers are seeded only from the new reset observation."""
        print("Testing: observation history reset clears previous episode data")
        env = ObservationHistoryWrapper(
            DummyHistoryObservationEnv(),
            history_length=3,
            entity_keys=("humans", "tables"),
        )
        env.reset()
        env.step(np.zeros(1, dtype=np.float32))

        reset_obs, _ = env.reset()

        np.testing.assert_array_equal(reset_obs["robot"][:, 0], [0.0, 0.0, 0.0])
        np.testing.assert_array_equal(reset_obs["humans"][0, :, 6], [0.0, 0.0, 0.0])

    def test_socnav_entity_counts_match_padded_observation_formula(self):
        """Testing: fixed observation counts use SocNavGym's padded entity formulas."""
        print("Testing: fixed observation counts match SocNavGym interaction-aware max formulas")
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "socnav.yaml"
            config_path.write_text(
                textwrap.dedent(
                    """
                    env:
                        get_padded_observations: true
                        max_static_humans: 3
                        max_dynamic_humans: 5
                        max_tables: 2
                        max_plants: 4
                        max_laptops: 6
                        max_h_h_dynamic_interactions: 2
                        max_h_h_dynamic_interactions_non_dispersing: 1
                        max_h_h_static_interactions: 3
                        max_h_h_static_interactions_non_dispersing: 1
                        max_human_in_h_h_interactions: 4
                        max_h_l_interactions: 2
                        max_h_l_interactions_non_dispersing: 1
                    """
                )
            )

            counts = derive_socnav_entity_counts(config_path, wall_count=8)

        self.assertEqual(counts["humans"], 39)
        self.assertEqual(counts["laptops"], 9)
        self.assertEqual(counts["tables"], 2)
        self.assertEqual(counts["plants"], 4)
        self.assertEqual(counts["walls"], 8)

    def test_fixed_observation_wrapper_pads_missing_and_short_entity_keys(self):
        """Testing: fixed observation wrapper pads entity keys deterministically."""
        print("Testing: fixed observation wrapper pads missing and short entity observations")
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "socnav.yaml"
            write_fixed_observation_target_config(
                config_path,
                {
                    "max_static_humans": 1,
                    "max_dynamic_humans": 2,
                    "max_laptops": 2,
                    "max_human_in_h_h_interactions": 4,
                    "max_h_l_interactions": 1,
                },
            )
            env = FixedObservationSpaceWrapper(
                DummyFixedObservationEnv(humans_rows=2),
                config_path=config_path,
                include_keys=("humans", "laptops", "tables"),
            )

            obs_a, _ = env.reset()
            obs_b, _ = env.reset()
            _, _, _, _, info = env.step(np.zeros(1, dtype=np.float32))

        self.assertEqual(env.observation_space.spaces["humans"].shape, (4 * 14,))
        self.assertEqual(env.observation_space.spaces["laptops"].shape, (3 * 14,))
        self.assertEqual(env.observation_space.spaces["tables"].shape, (0,))
        np.testing.assert_array_equal(obs_a["humans"], obs_b["humans"])
        np.testing.assert_array_equal(obs_a["humans"][:28], np.arange(28, dtype=np.float32))
        np.testing.assert_array_equal(obs_a["humans"][28:], np.zeros(28, dtype=np.float32))
        np.testing.assert_array_equal(obs_a["laptops"], np.zeros(42, dtype=np.float32))
        self.assertEqual(obs_a["tables"].shape, (0,))
        self.assertEqual(info["terminal_observation"]["humans"].shape, (56,))

    def test_fixed_observation_wrapper_uses_target_entity_bounds(self):
        """Testing: fixed observation wrapper derives native entity bounds from the target config."""
        print("Testing: fixed observation wrapper uses target config bounds")
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "socnav.yaml"
            write_fixed_observation_target_config(
                config_path,
                {
                    "max_static_humans": 1,
                    "max_dynamic_humans": 2,
                    "max_human_in_h_h_interactions": 4,
                },
            )
            env = FixedObservationSpaceWrapper(
                DummyFixedObservationEnv(humans_rows=1, bounded_humans=True),
                config_path=config_path,
                include_keys=("humans",),
            )

        human_space = env.observation_space.spaces["humans"]
        self.assertEqual(human_space.shape, (42,))
        np.testing.assert_allclose(human_space.low[:6], np.zeros(6, dtype=np.float32))
        np.testing.assert_allclose(human_space.high[:6], np.ones(6, dtype=np.float32))
        self.assertAlmostEqual(float(human_space.low[10]), -0.36)
        self.assertAlmostEqual(float(human_space.high[10]), 0.36)

    def test_architecture_entity_key_guard_uses_final_observation_space(self):
        """Testing: architecture entity keys must exist in PPO's final observation space."""
        print("Testing: architecture entity key guard checks final observation space")
        observation_space = spaces.Dict(
            {
                "robot": spaces.Box(low=-1.0, high=1.0, shape=(9,), dtype=np.float32),
                "humans": spaces.Box(low=-1.0, high=1.0, shape=(14,), dtype=np.float32),
                "tables": spaces.Box(low=-1.0, high=1.0, shape=(28,), dtype=np.float32),
            }
        )
        config = {"architecture": {"entity_keys": ["humans", "tables"]}}

        validate_architecture_entity_keys(config, observation_space)

        config["architecture"]["entity_keys"] = ["humans", "walls"]
        with self.assertRaisesRegex(ValueError, "missing from the final wrapped observation space"):
            validate_architecture_entity_keys(config, observation_space)

    def test_fixed_observation_wrapper_rejects_observations_over_capacity(self):
        """Testing: fixed observation wrapper raises if a stage exceeds the target capacity."""
        print("Testing: fixed observation wrapper rejects entity rows beyond fixed capacity")
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "socnav.yaml"
            write_fixed_observation_target_config(
                config_path,
                {
                    "max_static_humans": 1,
                    "max_human_in_h_h_interactions": 4,
                },
            )
            env = FixedObservationSpaceWrapper(
                DummyFixedObservationEnv(humans_rows=2),
                config_path=config_path,
                include_keys=("humans",),
            )

            with self.assertRaisesRegex(ValueError, "fixed observation capacity"):
                env.reset()

    def test_fixed_observation_wrapper_requires_padded_target_config(self):
        """Testing: fixed observation wrapper rejects non-padded target configs."""
        print("Testing: fixed observation wrapper requires padded SocNavGym target config")
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "socnav.yaml"
            config_path.write_text(
                textwrap.dedent(
                    """
                    env:
                        get_padded_observations: false
                        max_static_humans: 1
                        max_dynamic_humans: 0
                        max_tables: 0
                        max_plants: 0
                        max_laptops: 0
                        max_h_h_dynamic_interactions: 0
                        max_h_h_dynamic_interactions_non_dispersing: 0
                        max_h_h_static_interactions: 0
                        max_h_h_static_interactions_non_dispersing: 0
                        max_human_in_h_h_interactions: 4
                        max_h_l_interactions: 0
                        max_h_l_interactions_non_dispersing: 0
                    """
                )
            )

            with self.assertRaisesRegex(ValueError, "get_padded_observations: true"):
                FixedObservationSpaceWrapper(
                    DummyFixedObservationEnv(humans_rows=1),
                    config_path=config_path,
                    include_keys=("humans",),
                )

    def test_fixed_observation_wrapper_rejects_incomplete_target_bounds_config(self):
        """Testing: fixed observation wrapper does not fall back to live-env bounds."""
        print("Testing: fixed observation wrapper rejects incomplete target-bound configs")
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "socnav.yaml"
            config_path.write_text(
                textwrap.dedent(
                    """
                    env:
                        get_padded_observations: true
                        max_static_humans: 1
                        max_dynamic_humans: 0
                        max_tables: 0
                        max_plants: 0
                        max_laptops: 0
                        max_h_h_dynamic_interactions: 0
                        max_h_h_dynamic_interactions_non_dispersing: 0
                        max_h_h_static_interactions: 0
                        max_h_h_static_interactions_non_dispersing: 0
                        max_human_in_h_h_interactions: 0
                        max_h_l_interactions: 0
                        max_h_l_interactions_non_dispersing: 0
                    """
                )
            )

            with self.assertRaisesRegex(ValueError, "missing metadata"):
                FixedObservationSpaceWrapper(
                    DummyFixedObservationEnv(humans_rows=1, bounded_humans=True),
                    config_path=config_path,
                    include_keys=("humans",),
                )

    def test_fixed_observation_wrapper_requires_socnav_entity_feature_dim(self):
        """Testing: fixed observation wrapper rejects non-SocNavGym row widths."""
        print("Testing: fixed observation wrapper requires SocNavGym's 14-value entity rows")
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "socnav.yaml"
            write_fixed_observation_target_config(config_path, {"max_static_humans": 1})

            with self.assertRaisesRegex(ValueError, "entity_feature_dim: 14"):
                FixedObservationSpaceWrapper(
                    DummyFixedObservationEnv(humans_rows=1),
                    config_path=config_path,
                    include_keys=("humans",),
                    entity_feature_dim=15,
                )

    def test_fixed_observation_shapes_match_real_padded_socnavgym_shapes(self):
        """Testing: fixed wrapper shapes match a real padded SocNavGym target config."""
        print("Testing: fixed observation wrapper output shapes match real padded SocNavGym shapes")
        config = load_yaml(str(ROOT / "env_configs" / "env_main.yaml"))
        env_cfg = config["env"]
        env_cfg["get_padded_observations"] = True
        env_cfg["min_tables"] = env_cfg["max_tables"] = 1
        env_cfg["min_plants"] = env_cfg["max_plants"] = 1
        env_cfg["min_laptops"] = env_cfg["max_laptops"] = 1
        env_cfg["min_h_h_static_interactions"] = env_cfg["max_h_h_static_interactions"] = 0

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "final_socnav.yaml"
            config_path.write_text(yaml.safe_dump(config, sort_keys=False))
            final_env = gym.make("SocNavGym-v1", config=str(config_path))
            wrapped_env = FixedObservationSpaceWrapper(
                DummyFixedObservationEnv(humans_rows=1),
                config_path=config_path,
                include_keys=("humans", "laptops", "tables", "plants"),
            )

            obs, _ = wrapped_env.reset()
            final_spaces = final_env.observation_space.spaces

            try:
                for key in ("humans", "laptops", "tables", "plants"):
                    self.assertEqual(wrapped_env.observation_space.spaces[key].shape, final_spaces[key].shape)
                    self.assertEqual(obs[key].shape, final_spaces[key].shape)
                    np.testing.assert_allclose(wrapped_env.observation_space.spaces[key].low, final_spaces[key].low)
                    np.testing.assert_allclose(wrapped_env.observation_space.spaces[key].high, final_spaces[key].high)
            finally:
                final_env.close()

    def test_architecture_feature_dims_match_expected_ppo_feature_size(self):
        """Testing: each architecture exposes its configured PPO feature size."""
        print("Testing: architecture feature dimensions for PPO feature extractor")
        configs = {
            "social_context_fusion": ROOT / "architectures" / "social_context_fusion" / "config.yaml",
            "feedforward_social_context_fusion": ROOT / "architectures" / "feedforward_social_context_fusion" / "config.yaml",
            "joint_pair_context_fusion": ROOT / "architectures" / "joint_pair_context_fusion" / "config.yaml",
            "hybrid_context_fusion": ROOT / "architectures" / "hybrid_context_fusion" / "config.yaml",
            "dual_context_fusion": ROOT / "architectures" / "dual_context_fusion" / "config.yaml",
            "joint_scene_fusion": ROOT / "architectures" / "joint_scene_fusion" / "config.yaml",
            "crowd_context_fusion": ROOT / "architectures" / "crowd_context_fusion" / "config.yaml",
        }
        expected_dims = {
            "social_context_fusion": 128,
            "feedforward_social_context_fusion": 128,
            "joint_pair_context_fusion": 128,
            "hybrid_context_fusion": 85,
            "dual_context_fusion": 130,
            "joint_scene_fusion": 128,
            "crowd_context_fusion": 128,
        }

        for name, path in configs.items():
            with self.subTest(name=name):
                config = load_yaml(str(path))
                self.assertEqual(architecture_feature_dim(name, config), expected_dims[name])

    def test_waypoint_features_extend_robot_input_before_architecture(self):
        """Testing: waypoint features are encoded through the robot branch, not appended after it."""
        print("Testing: waypoint features extend the robot input consumed by the architecture")
        observation_space = spaces.Dict(
            {
                "robot": spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32),
                "humans": spaces.Box(low=-np.inf, high=np.inf, shape=(14,), dtype=np.float32),
                "waypoint_features": spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32),
            }
        )

        extractor = ArchitectureFeaturesExtractor(
            observation_space,
            architecture_name="social_context_fusion",
            architecture_config_path=str(ROOT / "architectures" / "social_context_fusion" / "config.yaml"),
            entity_keys=("humans",),
            entity_feature_dim=14,
            include_waypoint_features=True,
        )

        expected_dim = architecture_feature_dim(
            "social_context_fusion",
            load_yaml(str(ROOT / "architectures" / "social_context_fusion" / "config.yaml")),
        )
        self.assertEqual(extractor.features_dim, expected_dim)
        self.assertEqual(effective_robot_input_dim(observation_space, include_waypoint_features=True), 17)
        self.assertEqual(extractor.base_robot_input_dim, 9)
        self.assertEqual(extractor.waypoint_input_dim, 8)
        self.assertEqual(extractor.effective_robot_input_dim, 17)
        self.assertEqual(extractor.architecture.robot_encoder.gru.input_size, 17)

        features = extractor(
            {
                "robot": torch.zeros((2, 9), dtype=torch.float32),
                "humans": torch.zeros((2, 14), dtype=torch.float32),
                "waypoint_features": torch.zeros((2, 8), dtype=torch.float32),
            }
        )

        self.assertEqual(tuple(features.shape), (2, expected_dim))

        matrix_waypoint_space = spaces.Dict(
            {
                "robot": spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32),
                "waypoint_features": spaces.Box(low=-np.inf, high=np.inf, shape=(2, 4), dtype=np.float32),
            }
        )
        self.assertEqual(effective_robot_input_dim(matrix_waypoint_space, include_waypoint_features=True), 17)

    def test_architecture_extractors_accept_temporal_observations(self):
        """Testing: every extractor accepts history-shaped pipeline observations."""
        print("Testing: all architecture extractors accept temporal robot, waypoint, and entity observations")
        env = ObservationHistoryWrapper(
            DummyHistoryObservationEnv(),
            history_length=3,
            entity_keys=("humans", "tables"),
        )
        obs, _ = env.reset()
        architecture_names = (
            "social_context_fusion",
            "feedforward_social_context_fusion",
            "joint_pair_context_fusion",
            "hybrid_context_fusion",
            "dual_context_fusion",
            "joint_scene_fusion",
            "crowd_context_fusion",
        )

        self.assertEqual(effective_robot_input_dim(env.observation_space, include_waypoint_features=True), 17)
        for name in architecture_names:
            with self.subTest(architecture=name):
                extractor = ArchitectureFeaturesExtractor(
                    env.observation_space,
                    architecture_name=name,
                    architecture_config_path=str(ROOT / "architectures" / name / "config.yaml"),
                    entity_keys=("humans", "tables"),
                    entity_feature_dim=14,
                    include_waypoint_features=True,
                )
                features = extractor({key: torch.as_tensor(value).unsqueeze(0) for key, value in obs.items()})

                self.assertEqual(tuple(features.shape), (1, extractor.features_dim))
                self.assertTrue(torch.isfinite(features).all())

    def test_csv_metric_writer_creates_reproducible_metric_file(self):
        """Testing: metric writer creates a CSV with stable headers and cleaned values."""
        print("Testing: CSV metric writer stores reproducible metric rows")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "metrics.csv"
            writer = CSVMetricWriter(path, ["episode", "SUCCESS", "PATH_LENGTH", "MISSING"])

            writer.write({"episode": 1, "SUCCESS": np.bool_(True), "PATH_LENGTH": np.float32(2.5)})

            content = path.read_text().splitlines()
            self.assertEqual(content[0], "episode,SUCCESS,PATH_LENGTH,MISSING")
            self.assertEqual(content[1], "1,True,2.5,")

    def test_run_directory_snapshots_reward_config_only_once(self):
        """Testing: resumed runs preserve the reward configuration captured at creation."""
        print("Testing: run directory snapshots reward config only once")
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "training_config.yaml"
            config_path.write_text("experiment: {}\n")
            config = {
                "experiment": {
                    "name": "snapshot_test",
                    "output_root": tmpdir,
                    "run_id": "run",
                    "resume_run_dir": None,
                    "copy_config": True,
                }
            }

            run_dir = make_run_dir(config, str(config_path))
            reward_snapshot = run_dir / REWARD_CONFIG_SNAPSHOT_NAME
            self.assertTrue(reward_snapshot.is_file())
            self.assertEqual(
                reward_snapshot.read_text(),
                (ROOT / "custom_rewards" / REWARD_CONFIG_SNAPSHOT_NAME).read_text(),
            )

            reward_snapshot.write_text("preserved reward settings\n")
            config["experiment"]["resume_run_dir"] = str(run_dir)
            make_run_dir(config, str(config_path))

            self.assertEqual(reward_snapshot.read_text(), "preserved reward settings\n")

    def test_evaluation_callback_uses_fixed_episode_seed_sequence(self):
        """Testing: evaluation episodes reuse a stable validation seed set."""
        print("Testing: evaluation callback maps episode numbers to fixed validation seeds")
        with tempfile.TemporaryDirectory() as tmpdir:
            callback = NavigationEvaluationCallback(
                eval_env=None,
                run_dir=Path(tmpdir),
                eval_interval_steps=100,
                n_eval_episodes=3,
                deterministic=True,
                navigation_csv_name="eval.csv",
                fixed_episode_seeds=True,
                eval_seed_base=10042,
            )

            self.assertEqual(callback._episode_seed(1), 10042)
            self.assertEqual(callback._episode_seed(2), 10043)
            self.assertEqual(callback._episode_seed(3), 10044)

            callback.fixed_episode_seeds = False
            self.assertIsNone(callback._episode_seed(1))

    def test_build_callbacks_uses_configured_eval_seed_base(self):
        """Testing: evaluation seed base is read directly from config."""
        print("Testing: callback builder uses configured eval seed base")
        config = {
            "training": {
                "total_timesteps": 1000,
                "checkpoint_interval_steps": 100,
                "reset_num_timesteps": True,
            },
            "metrics": {
                "eta_log_interval_steps": 100,
                "navigation_training_csv": "train.csv",
                "navigation_evaluation_csv": "eval.csv",
            },
            "evaluation": {
                "enabled": True,
                "eval_interval_steps": 100,
                "n_eval_episodes": 3,
                "deterministic": True,
                "fixed_episode_seeds": True,
                "eval_seed_base": 777,
            },
            "ppo": {"verbose": 0},
        }

        with tempfile.TemporaryDirectory() as tmpdir, patch("training_pipeline.train.make_eval_env", return_value=None):
            callbacks = build_callbacks(config, Path(tmpdir))

        eval_callback = next(callback for callback in callbacks.callbacks if isinstance(callback, NavigationEvaluationCallback))
        self.assertEqual(eval_callback.eval_seed_base, 777)

    def test_build_callbacks_adds_training_render_callback_when_enabled(self):
        """Testing: render_during_training wires a render callback."""
        print("Testing: training render callback is enabled from config")
        config = {
            "environment": {"render_during_training": True, "render_interval_steps": 3},
            "training": {
                "total_timesteps": 1000,
                "checkpoint_interval_steps": 100,
                "reset_num_timesteps": True,
            },
            "metrics": {
                "eta_log_interval_steps": 100,
                "navigation_training_csv": "train.csv",
                "navigation_evaluation_csv": "eval.csv",
            },
            "evaluation": {"enabled": False},
            "ppo": {"verbose": 0},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            callbacks = build_callbacks(config, Path(tmpdir))

        render_callback = next(callback for callback in callbacks.callbacks if isinstance(callback, TrainingRenderCallback))
        self.assertEqual(render_callback.render_interval_steps, 3)

    def test_training_time_sessions_accumulate_across_resumes(self):
        """Testing: training_time.json accumulates wall-clock duration across sessions."""
        print("Testing: training time sessions accumulate across resumed runs")
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)

            first = record_training_time_session(
                run_dir,
                started_at="2026-06-28T10:00:00+00:00",
                ended_at="2026-06-28T11:00:00+00:00",
                wall_clock_seconds=3600.0,
                start_timesteps=0,
                end_timesteps=1000,
            )
            second = record_training_time_session(
                run_dir,
                started_at="2026-06-29T10:00:00+00:00",
                ended_at="2026-06-29T10:30:00+00:00",
                wall_clock_seconds=1800.0,
                start_timesteps=1000,
                end_timesteps=1500,
            )

            self.assertEqual(first["total_wall_clock_seconds"], 3600.0)
            self.assertEqual(second["total_wall_clock_seconds"], 5400.0)
            self.assertEqual(len(second["sessions"]), 2)
            self.assertEqual(second["sessions"][1]["start_timesteps"], 1000)
            self.assertEqual(second["sessions"][1]["end_timesteps"], 1500)

    def test_training_time_session_updates_current_session_without_duplication(self):
        """Testing: running training-time session is updated, not duplicated."""
        print("Testing: training time session updates by session id")
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)

            running = record_training_time_session(
                run_dir,
                started_at="2026-06-28T10:00:00+00:00",
                ended_at="2026-06-28T10:10:00+00:00",
                wall_clock_seconds=600.0,
                start_timesteps=0,
                end_timesteps=500,
                session_id="session-a",
                status="running",
            )
            completed = record_training_time_session(
                run_dir,
                started_at="2026-06-28T10:00:00+00:00",
                ended_at="2026-06-28T10:30:00+00:00",
                wall_clock_seconds=1800.0,
                start_timesteps=0,
                end_timesteps=1000,
                session_id="session-a",
                status="completed",
            )

            self.assertEqual(len(running["sessions"]), 1)
            self.assertEqual(len(completed["sessions"]), 1)
            self.assertEqual(completed["total_wall_clock_seconds"], 1800.0)
            self.assertEqual(completed["sessions"][0]["status"], "completed")
            self.assertEqual(completed["sessions"][0]["end_timesteps"], 1000)

    def test_training_seed_history_appends_launch_seed_records(self):
        """Testing: training seed history records initial and resumed launch seeds."""
        print("Testing: training seed history records launch seeds")
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            first = {
                "experiment": {"seed": 42, "resume_run_dir": None},
                "training": {
                    "total_timesteps": 100000,
                    "resume_from_checkpoint": None,
                    "reset_num_timesteps": True,
                },
            }
            second = {
                "experiment": {"seed": 1042, "resume_run_dir": "runs/example"},
                "training": {
                    "total_timesteps": 100000,
                    "resume_from_checkpoint": "runs/example/checkpoints/ppo_step_100000.zip",
                    "reset_num_timesteps": False,
                },
            }

            record_training_seed_session(run_dir, first)
            data = record_training_seed_session(run_dir, second)

            self.assertEqual(len(data["sessions"]), 2)
            self.assertEqual(data["sessions"][0]["mode"], "initial")
            self.assertEqual(data["sessions"][0]["training_seed"], 42)
            self.assertEqual(data["sessions"][1]["mode"], "resume")
            self.assertEqual(data["sessions"][1]["training_seed"], 1042)
            self.assertEqual(data["sessions"][1]["resume_from_checkpoint"], second["training"]["resume_from_checkpoint"])
            self.assertFalse(data["sessions"][1]["reset_num_timesteps"])

    def test_training_callback_records_completed_time_after_final_save(self):
        """Testing: training callback writes completed timing through its end hook."""
        print("Testing: training callback records completed training time")
        with tempfile.TemporaryDirectory() as tmpdir:
            callback = NavigationTrainingCallback(
                run_dir=Path(tmpdir),
                total_timesteps=1500,
                checkpoint_interval_steps=0,
                eta_log_interval_steps=100,
                navigation_csv_name="train.csv",
            )
            model = DummyModel()
            callback.model = model
            callback.num_timesteps = 1000

            callback._on_training_start()
            callback.num_timesteps = 1500
            callback._on_training_end()

            self.assertTrue(model.saved_paths)
            self.assertIn("time/session_wall_clock_seconds", model.logger.records)
            path = Path(tmpdir) / "training_time.json"
            data = json.loads(path.read_text())
            self.assertEqual(len(data["sessions"]), 1)
            self.assertEqual(data["sessions"][0]["status"], "completed")
            self.assertEqual(data["sessions"][0]["start_timesteps"], 1000)
            self.assertEqual(data["sessions"][0]["end_timesteps"], 1500)

    def test_resumed_training_callback_uses_global_eta_target(self):
        """Testing: resumed ETA target includes checkpoint timesteps plus requested timesteps."""
        print("Testing: resumed training callback uses global ETA target")
        with tempfile.TemporaryDirectory() as tmpdir:
            callback = NavigationTrainingCallback(
                run_dir=Path(tmpdir),
                total_timesteps=100000,
                checkpoint_interval_steps=10000,
                eta_log_interval_steps=1000,
                navigation_csv_name="train.csv",
                reset_num_timesteps=False,
            )
            callback.model = DummyModel()
            callback.num_timesteps = 100352

            callback._on_training_start()
            callback.num_timesteps = 101353
            callback._maybe_log_eta()

            self.assertEqual(callback.total_timesteps, 200352)
            self.assertEqual(callback.last_checkpoint_step, 100352)
            self.assertEqual(callback.last_eta_step, 101353)
            self.assertGreater(callback.model.logger.records["time/eta_seconds"], 0.0)

    def test_training_progress_print_includes_last_episode_reward_and_steps(self):
        """Testing: ETA print includes the latest completed episode reward and length."""
        print("Testing: training progress print includes last episode reward and steps")
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            csv_path = run_dir / "metrics" / "train.csv"
            writer = CSVMetricWriter(csv_path, ["episode", "global_step", "episode_reward", "episode_length"])
            writer.write({"episode": 1, "global_step": 10, "episode_reward": 1.5, "episode_length": 4})
            writer.write({"episode": 2, "global_step": 20, "episode_reward": -0.5, "episode_length": 6})

            callback = NavigationTrainingCallback(
                run_dir=run_dir,
                total_timesteps=100,
                checkpoint_interval_steps=0,
                eta_log_interval_steps=10,
                navigation_csv_name="train.csv",
                verbose=1,
            )
            callback.model = DummyModel()
            callback.num_timesteps = 20
            callback._on_training_start()
            callback.num_timesteps = 30
            callback.locals = {
                "dones": [True],
                "infos": [{"episode": {"r": 2.25, "l": 5}}],
                "rewards": [0.0],
            }

            buffer = io.StringIO()
            with redirect_stdout(buffer):
                callback._on_step()

            output = buffer.getvalue()
            self.assertIn("Training progress: 30/100 steps", output)
            self.assertIn("episode reward 2.25", output)
            self.assertIn("episode steps 5", output)
            self.assertAlmostEqual(callback.last_episode_reward, 2.25)
            self.assertEqual(callback.last_episode_steps, 5)
            self.assertEqual(callback.model.logger.records["navigation/train/last_episode_steps"], 5)

    def test_evaluation_callback_starts_interval_from_resume_step(self):
        """Testing: resumed evaluation waits one full eval interval before running."""
        print("Testing: resumed evaluation callback anchors interval at resume step")
        with tempfile.TemporaryDirectory() as tmpdir:
            callback = NavigationEvaluationCallback(
                eval_env=None,
                run_dir=Path(tmpdir),
                eval_interval_steps=20000,
                n_eval_episodes=5,
                deterministic=True,
                navigation_csv_name="eval.csv",
                fixed_episode_seeds=True,
                eval_seed_base=10042,
            )
            callback.num_timesteps = 100352

            callback._on_training_start()

            self.assertEqual(callback.last_eval_step, 100352)


if __name__ == "__main__":
    unittest.main()
