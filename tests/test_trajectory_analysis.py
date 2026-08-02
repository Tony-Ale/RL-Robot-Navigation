import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from environment_inspection.trajectory_analysis import collect_trajectory, run_trajectory_analysis


class _Robot:
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.goal_x = 2.0
        self.goal_y = 0.0


class _ActionSpace:
    def sample(self):
        return np.array([0.0, 0.0], dtype=np.float32)


class _TrajectoryEnv:
    TIMESTEP = 0.25
    MAP_X = 10.0
    MAP_Y = 8.0

    def __init__(self):
        self.robot = _Robot()
        self.action_space = _ActionSpace()
        self.latest_plan = None
        self.episode_astar_path_length = None
        self.steps = 0

    @property
    def unwrapped(self):
        return self

    def reset(self, seed=None):
        self.robot.x = 0.0
        self.robot.y = 0.0
        self.steps = 0
        self.latest_plan = SimpleNamespace(path_world=[(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)])
        self.episode_astar_path_length = 2.0
        return {"robot": np.zeros(1)}, {}

    def step(self, action):
        self.steps += 1
        self.robot.x += 1.0
        terminated = self.steps == 2
        info = {}
        if terminated:
            info = {
                "SUCCESS": True,
                "COLLISION": False,
                "TIMEOUT": False,
                "PATH_LENGTH": 2.0,
                "A_STAR_PATH_LENGTH": 2.0,
                "A_STAR_SPL": 1.0,
            }
        return {"robot": np.zeros(1)}, 1.0, terminated, False, info

    def close(self):
        pass


class _FeedforwardPolicy:
    def predict(self, observation, env):
        return np.array([0.0, 0.0], dtype=np.float32)


class _StatefulPolicy(_FeedforwardPolicy):
    def __init__(self):
        self.reset_count = 0

    def reset(self):
        self.reset_count += 1


class TrajectoryAnalysisTests(unittest.TestCase):
    @patch(
        "environment_inspection.trajectory_analysis.render_environment_frame",
        return_value=np.zeros((20, 20, 3), dtype=np.uint8),
    )
    def test_stateful_trace_resets_memory_and_records_exact_positions(self, _render):
        env = _TrajectoryEnv()
        policy = _StatefulPolicy()

        trace = collect_trajectory(
            env,
            policy,
            seed=42,
            max_steps=10,
            require_astar=True,
            policy_type="stateful_ppo",
            checkpoint="checkpoint.zip",
        )

        self.assertEqual(policy.reset_count, 1)
        self.assertEqual(trace["robot_path"], [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)])
        self.assertEqual(trace["astar_path"], [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)])
        self.assertEqual(trace["summary"]["outcome"], "success")
        self.assertEqual(trace["summary"]["recorded_path_length"], 2.0)
        self.assertEqual(trace["summary"]["a_star_spl"], 1.0)

    @patch(
        "environment_inspection.trajectory_analysis.render_environment_frame",
        return_value=np.zeros((20, 20, 3), dtype=np.uint8),
    )
    def test_feedforward_trace_does_not_require_policy_state(self, _render):
        trace = collect_trajectory(
            _TrajectoryEnv(),
            _FeedforwardPolicy(),
            seed=7,
            max_steps=10,
            require_astar=True,
            policy_type="ppo",
        )

        self.assertEqual(trace["summary"]["policy_type"], "ppo")
        self.assertEqual(trace["summary"]["steps"], 2)

    @patch(
        "environment_inspection.trajectory_analysis.render_environment_frame",
        return_value=np.zeros((20, 20, 3), dtype=np.uint8),
    )
    def test_missing_required_astar_plan_is_rejected(self, _render):
        env = _TrajectoryEnv()
        original_reset = env.reset

        def reset_without_plan(seed=None):
            result = original_reset(seed=seed)
            env.latest_plan = None
            return result

        env.reset = reset_without_plan
        with self.assertRaisesRegex(RuntimeError, "requires a reset-time A\\* path"):
            collect_trajectory(
                env,
                _FeedforwardPolicy(),
                seed=7,
                max_steps=10,
                require_astar=True,
                policy_type="ppo",
            )

    @patch(
        "environment_inspection.trajectory_analysis.render_environment_frame",
        return_value=np.zeros((20, 20, 3), dtype=np.uint8),
    )
    @patch("environment_inspection.trajectory_analysis.save_trajectory_snapshot")
    def test_analysis_writes_seed_labeled_outputs(self, snapshot, _render):
        with tempfile.TemporaryDirectory() as directory:
            config = {
                "policy": {"type": "ppo", "checkpoint": None},
                "trajectory_analysis": {
                    "seeds": [42],
                    "max_steps": 10,
                    "require_astar": True,
                    "output_dir": directory,
                },
            }

            output_dir = run_trajectory_analysis(_TrajectoryEnv(), _FeedforwardPolicy(), config)

            episode_dir = Path(output_dir) / "seed_42_success"
            self.assertTrue((episode_dir / "trajectory.csv").is_file())
            self.assertTrue((episode_dir / "summary.json").is_file())
            self.assertTrue((Path(output_dir) / "trajectory_summary.csv").is_file())
            snapshot.assert_called_once()


if __name__ == "__main__":
    unittest.main()
