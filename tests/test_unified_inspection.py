import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from environment_inspection.inspect_environment import _analysis_mode, _checkpoint_path, _make_policy
from training_pipeline.episode_runtime import reset_policy


class _ResettablePolicy:
    def __init__(self):
        self.reset_count = 0

    def reset(self):
        self.reset_count += 1


class _StatefulPolicy:
    model = object()


class UnifiedInspectionTests(unittest.TestCase):
    def test_multiple_analysis_modes_are_rejected(self):
        config = {
            "failure_analysis": {"enabled": True},
            "stall_analysis": {"enabled": True},
            "wall_segment_analysis": {"enabled": False},
            "trajectory_analysis": {"enabled": False},
        }

        with self.assertRaisesRegex(ValueError, "Only one environment inspection analysis mode"):
            _analysis_mode(config)

    def test_single_analysis_mode_is_selected(self):
        config = {
            "failure_analysis": {"enabled": False},
            "stall_analysis": {"enabled": True},
            "wall_segment_analysis": {"enabled": False},
            "trajectory_analysis": {"enabled": False},
        }

        self.assertEqual(_analysis_mode(config), "stall_analysis")

    def test_trajectory_analysis_is_a_unified_inspection_mode(self):
        config = {
            "failure_analysis": {"enabled": False},
            "stall_analysis": {"enabled": False},
            "wall_segment_analysis": {"enabled": False},
            "trajectory_analysis": {"enabled": True},
        }

        self.assertEqual(_analysis_mode(config), "trajectory_analysis")

    def test_reset_policy_resets_stateful_policy_only_when_supported(self):
        policy = _ResettablePolicy()

        reset_policy(policy)
        reset_policy(None)

        self.assertEqual(policy.reset_count, 1)

    def test_stateful_loader_receives_inspection_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.zip"
            checkpoint.touch()
            env = object()
            policy = _StatefulPolicy()
            config = {
                "policy": {
                    "type": "stateful_ppo",
                    "checkpoint": str(checkpoint),
                    "deterministic": False,
                }
            }

            with (
                patch(
                    "stateful_training_pipeline.policies.load_stateful_policy",
                    return_value=policy,
                ) as loader,
                patch("environment_inspection.inspect_environment._validate_policy_spaces") as validate,
            ):
                loaded = _make_policy(config, env)

            self.assertIs(loaded, policy)
            loader.assert_called_once_with(
                checkpoint.resolve(),
                env=env,
                deterministic=False,
                device="auto",
            )
            validate.assert_called_once_with(policy.model, env)

    def test_missing_checkpoint_fails_before_model_loading(self):
        with self.assertRaisesRegex(FileNotFoundError, "checkpoint does not exist"):
            _checkpoint_path("missing-checkpoint.zip", "stateful_ppo")


if __name__ == "__main__":
    unittest.main()
