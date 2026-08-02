import io
import unittest
from contextlib import redirect_stdout

import numpy as np
from gym import spaces

from environment_inspection.failure_analysis import failure_type, initial_spawn_clearance
from environment_inspection.inspect_environment import (
    _empty_reward_totals,
    _entity_rows,
    _prepare_pipeline_config,
    _print_reward_breakdown,
    _print_observation_summary,
    _update_reward_totals,
    _validate_policy_spaces,
)


class DummyEntity:
    def __init__(self, name, x, y, **kwargs):
        self.name = name
        self.x = x
        self.y = y
        self.orientation = kwargs.get("orientation", 0.0)
        self.radius = kwargs.get("radius")
        self.width = kwargs.get("width")
        self.length = kwargs.get("length")
        self.thickness = kwargs.get("thickness")


class DummyBaseEnv:
    def __init__(self):
        self.robot = DummyEntity("robot", 0.0, 0.0, radius=0.25)
        self.static_humans = [DummyEntity("human", 1.0, 0.0, width=0.72)]
        self.dynamic_humans = []
        self.plants = [DummyEntity("plant", 2.0, 0.0, radius=0.4)]
        self.tables = []
        self.laptops = []
        self.walls = [DummyEntity("wall", 0.0, 1.0, length=2.0, thickness=0.2)]
        self.moving_interactions = []
        self.static_interactions = []
        self.h_l_interactions = []


class DummyWrappedEnv:
    def __init__(self, base_env):
        self.unwrapped = base_env


class TestEnvironmentInspection(unittest.TestCase):
    def test_initial_spawn_clearance_uses_nearest_surface_distance(self):
        """Testing: spawn clearance uses geometry surfaces, not center distance."""
        print("Testing: initial spawn clearance uses nearest entity surface")
        clearance = initial_spawn_clearance(DummyWrappedEnv(DummyBaseEnv()))

        self.assertEqual(clearance["entity_type"], "human")
        self.assertAlmostEqual(clearance["clearance"], 0.39, places=6)

    def test_failure_type_prefers_specific_collision_flags(self):
        """Testing: failure analysis labels specific failure causes."""
        print("Testing: failure analysis failure type classification")

        self.assertEqual(failure_type({"COLLISION_HUMAN": True}, True, False), "collision_human")
        self.assertEqual(failure_type({"COLLISION_WALL": True}, True, False), "collision_wall")
        self.assertEqual(failure_type({"TIMEOUT": True}, False, True), "timeout")
        self.assertEqual(failure_type({}, False, False), "max_steps")

    def test_temporal_entity_summary_uses_latest_frame(self):
        """Testing: history frames are not counted as additional entities."""
        humans = np.zeros((2, 3, 14), dtype=np.float32)
        humans[0, :, 1] = 1.0
        humans[1, :-1, 1] = 1.0
        config = {"debug": {"entity_keys": ["humans"], "entity_feature_dim": 14}}

        rows = _entity_rows(humans, 14)
        output = io.StringIO()
        with redirect_stdout(output):
            _print_observation_summary({"humans": humans}, config, verbosity=3)

        self.assertEqual(rows.shape, (2, 14))
        self.assertEqual(rows[1].sum(), 0.0)
        self.assertIn("humans: real=1, padded=1, capacity=2, history=3", output.getvalue())
        self.assertIn("latest_rows=", output.getvalue())

    def test_policy_space_mismatch_has_inspection_specific_error(self):
        """Testing: incompatible checkpoints fail before the first prediction."""
        class Model:
            observation_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
            action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)

        class Env:
            observation_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
            action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)

        with self.assertRaisesRegex(ValueError, "saved training config"):
            _validate_policy_spaces(Model(), Env())

    def test_static_warning_reward_has_separate_accurate_total(self):
        totals = _empty_reward_totals()
        info = {
            "reward_reason": "shaped",
            "warning_zone_reward": -0.1,
            "static_warning_zone_reward": -0.2,
            "checkpoint_reward": 0.3,
            "stagnation_penalty": -0.02,
        }

        _update_reward_totals(totals, reward=0.38, info=info)

        self.assertAlmostEqual(totals["progress_reward"], 0.4)
        self.assertAlmostEqual(totals["warning_zone_reward"], -0.1)
        self.assertAlmostEqual(totals["static_warning_zone_reward"], -0.2)

    def test_reward_print_groups_static_diagnostics_cleanly(self):
        totals = _empty_reward_totals()
        info = {
            "reward_reason": "shaped",
            "static_warning_zone_reward": -0.03,
            "static_warning_zone_hits": 1,
            "nearest_static_clearance": 0.12,
            "nearest_static_type": "wall",
        }
        output = io.StringIO()

        with redirect_stdout(output):
            _print_reward_breakdown(info, totals, verbosity=2)

        text = output.getvalue()
        self.assertIn("components: static_warning_zone_reward=-0.030000", text)
        self.assertIn("safety: static_warning_zone_hits=1", text)
        self.assertIn("nearest_static_clearance=0.120000", text)
        self.assertIn("nearest_static_type=wall", text)

    def test_inspection_warning_zone_override_does_not_mutate_pipeline_config(self):
        pipeline_config = {
            "wrappers": {
                "warning_zone_visualization": {
                    "enabled": False,
                    "config_path": "warning.yaml",
                }
            }
        }

        prepared = _prepare_pipeline_config(
            {"visualization": {"warning_zones": True}},
            pipeline_config,
        )

        self.assertTrue(prepared["wrappers"]["warning_zone_visualization"]["enabled"])
        self.assertEqual(
            prepared["wrappers"]["warning_zone_visualization"]["config_path"],
            "warning.yaml",
        )
        self.assertFalse(pipeline_config["wrappers"]["warning_zone_visualization"]["enabled"])


if __name__ == "__main__":
    unittest.main()
