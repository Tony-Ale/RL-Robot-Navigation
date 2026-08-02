from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from gym import spaces

from testing_pipeline.metrics import paired_comparison_rows, summarize_comparison, summarize_rows
from testing_pipeline.evaluate_checkpoints import (
    checkpoint_step,
    evaluated_checkpoints,
    format_seconds,
    offline_seeds,
    resolve_checkpoints,
    summary_row,
)
from testing_pipeline.policies import ORCARobotPolicy, orca_velocity_to_action
from testing_pipeline.policy_loading import load_learned_agent_policy, resolve_policy_type
from testing_pipeline.render_agent import (
    RenderedVideoRecorder,
    _current_render_frame,
    _render_seeds,
    prepare_render_config,
    render_agent,
    render_from_config,
)
from testing_pipeline.runner import (
    clear_testing_outputs,
    resolve_checkpoint_path,
    run_policy_episodes,
    test_seeds,
    validate_testing_config,
)


class DummyRobot:
    def __init__(self):
        self.type = "diff-drive"
        self.orientation = 0.0


class DummyORCAEnv:
    def __init__(self):
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.robot = DummyRobot()
        self.MAX_ADVANCE_ROBOT = 0.5
        self.MAX_ROTATION = 1.0
        self.TIMESTEP = 1.0

    def compute_orca_velocity_robot(self, robot):
        return np.array([0.25, 0.0], dtype=np.float32)


class DummyOneStepEnv:
    def __init__(self):
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.reset_seeds = []

    def reset(self, seed=None):
        self.reset_seeds.append(seed)
        return np.zeros(1, dtype=np.float32), {}

    def step(self, action):
        info = {
            "SUCCESS": True,
            "COLLISION": False,
            "TIMEOUT": False,
            "PATH_LENGTH": 2.0,
            "A_STAR_PATH_LENGTH": 1.5,
            "A_STAR_SPL": 0.75,
            "SPL": 0.75,
            "STL": 0.5,
        }
        return np.zeros(1, dtype=np.float32), 1.0, True, False, info


class DummyPlannerFailEnv(DummyOneStepEnv):
    def __init__(self, failing_seeds):
        super().__init__()
        self.failing_seeds = set(failing_seeds)

    def reset(self, seed=None):
        self.reset_seeds.append(seed)
        if seed in self.failing_seeds:
            raise RuntimeError("Planner produced no waypoints after 1 reset attempt(s) with unchanged reset arguments.")
        return np.zeros(1, dtype=np.float32), {}


class DummyPolicy:
    controller_name = "learned_agent"

    def __init__(self):
        self.reset_calls = 0

    def reset(self):
        self.reset_calls += 1

    def predict(self, observation, env=None):
        return np.array([0.0, 0.0], dtype=np.float32)


class TestTestingPipeline(unittest.TestCase):
    def test_test_seed_generation_uses_held_out_offset(self):
        """Testing: final test episodes use reproducible held-out seeds."""
        print("Testing: final testing seed generation")
        config = {"experiment": {"seed": 42}}
        seeds = test_seeds(config, {"n_test_episodes": 3, "test_seed_offset": 20000, "fixed_test_seeds": True})

        self.assertEqual(seeds, [20042, 20043, 20044])

    def test_render_config_can_enable_warning_zone_overlay_without_mutating_source(self):
        """Testing: render helper enables warning overlays only in copied config."""
        print("Testing: render helper prepares warning-zone visualization config")
        config = {"wrappers": {"warning_zone_visualization": {"enabled": False}}}

        prepared = prepare_render_config(config, enable_warning_zones=True)

        self.assertTrue(prepared["wrappers"]["warning_zone_visualization"]["enabled"])
        self.assertFalse(config["wrappers"]["warning_zone_visualization"]["enabled"])

    def test_render_config_can_enable_path_waypoint_overlay_without_mutating_source(self):
        """Testing: render helper enables A* path overlays only in copied config."""
        print("Testing: render helper prepares A* path and waypoint overlay config")
        config = {"wrappers": {"astar": {"enabled": False}, "navigation_features": {"enabled": True}}}

        prepared = prepare_render_config(config, enable_path_waypoints=True)

        self.assertTrue(prepared["wrappers"]["astar"]["enabled"])
        self.assertTrue(prepared["wrappers"]["navigation_features"]["config"]["visualization"]["enabled"])
        self.assertFalse(config["wrappers"]["astar"]["enabled"])
        self.assertNotIn("config", config["wrappers"]["navigation_features"])

    def test_render_seed_helper_uses_explicit_seed_sequence(self):
        """Testing: render helper uses explicit seed sequence when provided."""
        print("Testing: render helper builds explicit seed sequence")
        config = {"experiment": {"seed": 42}, "testing": {"test_seed_offset": 20000, "fixed_test_seeds": True}}

        self.assertEqual(_render_seeds(config, seed=7, episodes=3), [7, 8, 9])
        self.assertEqual(_render_seeds(config, seed=None, episodes=2), [20042, 20043])

    def test_renderer_reads_all_runtime_settings_from_yaml(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            training_config = root / "training.yaml"
            training_config.write_text("architecture:\n  name: stateful_social_context_fusion\n")
            render_config = root / "render.yaml"
            render_config.write_text(
                "rendering:\n"
                f"  training_config_path: {training_config}\n"
                f"  run_dir: {root / 'run'}\n"
                "  checkpoint_path: checkpoint.zip\n"
                "  policy_type: stateful_ppo\n"
                "  device: cpu\n"
                "  seed: 42\n"
                "  episodes: 3\n"
                "  deterministic: false\n"
                "  delay_seconds: 0.1\n"
                "  warning_zones: true\n"
                "  path_waypoints: true\n"
                "  record_video: true\n"
                f"  video_path: {root / 'agent.mp4'}\n"
                "  video_fps: 12.5\n"
            )

            with patch("testing_pipeline.render_agent.render_agent") as render:
                render_from_config(str(render_config))

        render.assert_called_once_with(
            config={"architecture": {"name": "stateful_social_context_fusion"}},
            run_dir=root / "run",
            checkpoint_path=Path("checkpoint.zip"),
            seed=42,
            episodes=3,
            deterministic=False,
            delay_seconds=0.1,
            enable_warning_zones=True,
            enable_path_waypoints=True,
            policy_type="stateful_ppo",
            device="cpu",
            video_path=root / "agent.mp4",
            video_fps=12.5,
        )

    def test_renderer_requires_video_path_when_recording_is_enabled(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            training_config = root / "training.yaml"
            training_config.write_text("architecture:\n  name: feedforward_social_context_fusion\n")
            render_config = root / "render.yaml"
            render_config.write_text(
                "rendering:\n"
                f"  training_config_path: {training_config}\n"
                f"  run_dir: {root / 'run'}\n"
                "  record_video: true\n"
            )

            with self.assertRaisesRegex(ValueError, "video_path is required"):
                render_from_config(str(render_config))

    def test_video_recorder_writes_rendered_frame_and_releases_writer(self):
        writer = MagicMock()
        writer.isOpened.return_value = True
        frame = np.zeros((12, 20, 3), dtype=np.uint8)

        with tempfile.TemporaryDirectory() as tmpdir, patch(
            "testing_pipeline.render_agent.cv2.VideoWriter_fourcc", return_value=1234
        ) as fourcc, patch(
            "testing_pipeline.render_agent.cv2.VideoWriter", return_value=writer
        ) as video_writer:
            output_path = Path(tmpdir) / "agent.mp4"
            recorder = RenderedVideoRecorder(output_path, fps=8.0)
            recorder.write(frame)
            recorder.close()

        fourcc.assert_called_once_with(*"mp4v")
        video_writer.assert_called_once_with(str(output_path), 1234, 8.0, (20, 12))
        writer.write.assert_called_once_with(frame)
        writer.release.assert_called_once_with()

    def test_video_capture_copies_completed_base_environment_frame(self):
        frame = np.full((4, 5, 3), 17, dtype=np.uint8)
        base_env = type("BaseEnv", (), {"world_image": frame})()
        wrapped_env = type("WrappedEnv", (), {"unwrapped": base_env})()

        captured = _current_render_frame(wrapped_env)
        frame[0, 0, 0] = 99

        self.assertEqual(captured.shape, (4, 5, 3))
        self.assertEqual(int(captured[0, 0, 0]), 17)

    def test_render_loop_uses_shared_loader_and_resets_each_episode(self):
        class RenderEnv(DummyOneStepEnv):
            def __init__(self):
                super().__init__()
                self.closed = False

            def render(self):
                return None

            def close(self):
                self.closed = True

        env = RenderEnv()
        policy = DummyPolicy()
        config = {"experiment": {"seed": 42}, "testing": {}}
        with patch("testing_pipeline.render_agent.make_eval_env", return_value=env), patch(
            "testing_pipeline.render_agent.load_learned_agent_policy",
            return_value=policy,
        ) as loader:
            render_agent(
                config=config,
                run_dir=Path("run"),
                checkpoint_path=Path("checkpoint.zip"),
                seed=10,
                episodes=2,
                policy_type="stateful_ppo",
                device="cpu",
            )

        self.assertEqual(policy.reset_calls, 2)
        self.assertEqual(env.reset_seeds, [10, 11])
        self.assertTrue(env.closed)
        loader.assert_called_once()
        loader_args = loader.call_args.args
        self.assertEqual(loader_args[:3], (
            Path("checkpoint.zip"),
            env,
            {"deterministic": True, "device": "cpu", "policy_type": "stateful_ppo"},
        ))
        self.assertEqual(loader_args[3]["experiment"], config["experiment"])

    def test_run_policy_episodes_records_rows_and_csv(self):
        """Testing: policy episode runner writes learned-agent metrics."""
        print("Testing: final testing episode runner writes metric rows")
        with tempfile.TemporaryDirectory() as tmpdir:
            env = DummyOneStepEnv()
            policy = DummyPolicy()
            rows = run_policy_episodes(
                env=env,
                policy=policy,
                seeds=[10, 11],
                checkpoint_path=Path("model.zip"),
                csv_path=Path(tmpdir) / "agent.csv",
            )

            self.assertEqual(env.reset_seeds, [10, 11])
            self.assertEqual(policy.reset_calls, 2)
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["controller"], "learned_agent")
            self.assertEqual(rows[0]["SUCCESS"], True)
            content = (Path(tmpdir) / "agent.csv").read_text().splitlines()
            self.assertIn("controller", content[0])
            self.assertIn("A_STAR_SPL", content[0])
            self.assertEqual(len(content), 3)

    def test_run_policy_episodes_skips_planner_failed_seed_until_valid_count(self):
        """Testing: final testing keeps scanning seeds until it records valid episodes."""
        print("Testing: final testing skips no-waypoint seeds and fills valid episode count")
        with tempfile.TemporaryDirectory() as tmpdir:
            env = DummyPlannerFailEnv(failing_seeds={10})
            rows = run_policy_episodes(
                env=env,
                policy=DummyPolicy(),
                seeds=[10, 11],
                checkpoint_path=Path("model.zip"),
                csv_path=Path(tmpdir) / "agent.csv",
            )

            self.assertEqual([row["seed"] for row in rows], [11, 12])
            self.assertEqual(len(rows), 2)
            self.assertEqual(env.reset_seeds, [10, 11, 12])

    def test_policy_type_is_explicit_or_derived_from_architecture(self):
        self.assertEqual(resolve_policy_type({"policy_type": "ppo"}), "ppo")
        self.assertEqual(resolve_policy_type({"policy_type": "stateful_ppo"}), "stateful_ppo")
        self.assertEqual(
            resolve_policy_type({}, {"architecture": {"name": "stateful_social_context_fusion"}}),
            "stateful_ppo",
        )
        self.assertEqual(
            resolve_policy_type({}, {"architecture": {"name": "feedforward_social_context_fusion"}}),
            "ppo",
        )
        with self.assertRaisesRegex(ValueError, "policy_type"):
            resolve_policy_type({"policy_type": "unknown"})

    def test_unified_loader_dispatches_stateful_checkpoint(self):
        expected = object()
        with patch(
            "stateful_training_pipeline.policies.load_stateful_policy",
            return_value=expected,
        ) as loader:
            loaded = load_learned_agent_policy(
                Path("checkpoint.zip"),
                env="env",
                settings={"policy_type": "stateful_ppo", "deterministic": False, "device": "cpu"},
            )

        self.assertIs(loaded, expected)
        loader.assert_called_once_with(
            Path("checkpoint.zip"),
            env="env",
            deterministic=False,
            device="cpu",
        )

    def test_shared_episode_loop_resets_stateful_memory_between_episodes(self):
        from stateful_training_pipeline.policies import StatefulLearnedAgentPolicy

        class Model:
            def __init__(self):
                self.inputs = []

            def predict(self, observation, state, episode_start, deterministic):
                self.inputs.append((state, bool(episode_start[0])))
                return np.zeros((2,), dtype=np.float32), len(self.inputs)

        model = Model()
        run_policy_episodes(
            env=DummyOneStepEnv(),
            policy=StatefulLearnedAgentPolicy(model),
            seeds=[10, 11],
            checkpoint_path=Path("stateful.zip"),
            csv_path=None,
        )

        self.assertEqual(model.inputs, [(None, True), (None, True)])

    def test_checkpoint_resolution_uses_largest_final_step(self):
        """Testing: final testing picks the largest numeric final checkpoint."""
        print("Testing: final testing checkpoint resolution uses numeric step order")
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir) / "checkpoints"
            checkpoint_dir.mkdir()
            (checkpoint_dir / "ppo_final_step_900.zip").write_text("")
            (checkpoint_dir / "ppo_final_step_1000.zip").write_text("")

            checkpoint = resolve_checkpoint_path({}, Path(tmpdir))

            self.assertEqual(checkpoint.name, "ppo_final_step_1000.zip")

    def test_agent_orca_comparison_is_paired_by_seed(self):
        """Testing: learned-agent and ORCA rows are compared by matching seed."""
        print("Testing: learned-agent vs ORCA comparison rows")
        agent_rows = [
            {"seed": 2, "SUCCESS": True, "COLLISION": False, "A_STAR_PATH_LENGTH": 3.0, "A_STAR_SPL": 0.75, "SPL": 0.8, "STL": 0.7, "PATH_LENGTH": 4.0},
            {"seed": 1, "SUCCESS": False, "COLLISION": True, "A_STAR_PATH_LENGTH": 4.0, "A_STAR_SPL": 0.0, "SPL": 0.0, "STL": 0.0, "PATH_LENGTH": 8.0},
        ]
        orca_rows = [
            {"seed": 1, "SUCCESS": True, "COLLISION": False, "A_STAR_PATH_LENGTH": 4.0, "A_STAR_SPL": 0.8, "SPL": 0.5, "STL": 0.4, "PATH_LENGTH": 5.0},
            {"seed": 2, "SUCCESS": True, "COLLISION": False, "A_STAR_PATH_LENGTH": 3.0, "A_STAR_SPL": 0.5, "SPL": 0.6, "STL": 0.6, "PATH_LENGTH": 6.0},
        ]

        rows = paired_comparison_rows(agent_rows, orca_rows)

        self.assertEqual([row["seed"] for row in rows], [1, 2])
        self.assertEqual(rows[0]["winner"], "orca")
        self.assertEqual(rows[1]["winner"], "agent")
        self.assertEqual(rows[1]["agent_A_STAR_PATH_LENGTH"], rows[1]["orca_A_STAR_PATH_LENGTH"])
        self.assertEqual(rows[1]["delta_A_STAR_SPL"], 0.25)
        self.assertEqual(rows[1]["delta_SPL"], 0.20000000000000007)

    def test_comparison_requires_fixed_seeds(self):
        """Testing: ORCA comparison rejects unpaired non-seeded test episodes."""
        print("Testing: ORCA comparison requires fixed test seeds")
        with self.assertRaises(ValueError):
            validate_testing_config({"compare_with_baseline": True, "fixed_test_seeds": False})

    def test_missing_success_metric_makes_winner_unknown(self):
        """Testing: missing SUCCESS metrics are not treated as failed episodes."""
        print("Testing: missing SUCCESS metrics produce unknown comparison winner")
        rows = paired_comparison_rows(
            [{"seed": 1, "SPL": 1.0}],
            [{"seed": 1, "SUCCESS": True, "SPL": 0.5}],
        )

        self.assertEqual(rows[0]["winner"], "unknown")

    def test_missing_astar_spl_makes_winner_unknown(self):
        """Testing: new comparisons require the configured A*-referenced metric."""
        print("Testing: missing A*-SPL produces unknown comparison winner")
        rows = paired_comparison_rows(
            [{"seed": 1, "SUCCESS": True, "SPL": 1.0}],
            [{"seed": 1, "SUCCESS": True, "SPL": 0.5}],
        )

        self.assertEqual(rows[0]["winner"], "unknown")

    def test_summary_metrics_include_rates_and_comparison_counts(self):
        """Testing: final testing summary reports aggregate metrics."""
        print("Testing: final testing aggregate summaries")
        rows = [
            {"SUCCESS": True, "COLLISION": False, "TIMEOUT": False, "A_STAR_SPL": 0.8, "SPL": 0.8, "episode_length": 10},
            {"SUCCESS": False, "COLLISION": True, "TIMEOUT": False, "A_STAR_SPL": 0.0, "SPL": 0.0, "episode_length": 20},
        ]
        comparison_rows = [
            {"winner": "agent", "delta_A_STAR_SPL": 0.2, "delta_SPL": 0.2},
            {"winner": "orca", "delta_A_STAR_SPL": -0.1, "delta_SPL": -0.1},
        ]

        summary = summarize_rows(rows)
        comparison = summarize_comparison(comparison_rows)

        self.assertEqual(summary["success_rate"], 0.5)
        self.assertEqual(summary["collision_rate"], 0.5)
        self.assertEqual(summary["mean_spl"], 0.4)
        self.assertEqual(summary["mean_a_star_spl"], 0.4)
        self.assertAlmostEqual(comparison["mean_delta_a_star_spl"], 0.05)
        self.assertEqual(comparison["agent_win_count"], 1)
        self.assertEqual(comparison["orca_win_count"], 1)
        self.assertEqual(comparison["unknown_count"], 0)

    def test_offline_checkpoint_step_parses_step_and_final_names(self):
        """Testing: offline evaluator extracts numeric steps from checkpoint names."""
        print("Testing: offline checkpoint step parsing")

        self.assertEqual(checkpoint_step(Path("ppo_step_10000.zip")), 10000)
        self.assertEqual(checkpoint_step(Path("ppo_final_step_100352.zip")), 100352)

    def test_offline_checkpoint_resolution_supports_sources_and_numeric_sort(self):
        """Testing: offline evaluator resolves requested checkpoints in numeric order."""
        print("Testing: offline checkpoint source resolution")
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir) / "checkpoints"
            checkpoint_dir.mkdir()
            for name in ["ppo_step_200.zip", "ppo_step_100.zip", "ppo_final_step_150.zip"]:
                (checkpoint_dir / name).write_text("")

            all_names = [path.name for path in resolve_checkpoints(Path(tmpdir), {"checkpoint_source": "all"})]
            step_names = [path.name for path in resolve_checkpoints(Path(tmpdir), {"checkpoint_source": "step"})]
            listed_names = [
                path.name
                for path in resolve_checkpoints(
                    Path(tmpdir),
                    {"checkpoint_source": "list", "checkpoint_filenames": ["ppo_final_step_150.zip"]},
                )
            ]

            self.assertEqual(all_names, ["ppo_step_100.zip", "ppo_final_step_150.zip", "ppo_step_200.zip"])
            self.assertEqual(step_names, ["ppo_step_100.zip", "ppo_step_200.zip"])
            self.assertEqual(listed_names, ["ppo_final_step_150.zip"])

    def test_offline_seeds_default_to_hundred_fixed_episodes(self):
        """Testing: offline evaluator defaults to 100 fixed evaluation seeds."""
        print("Testing: offline evaluator seed defaults")

        seeds = offline_seeds({})

        self.assertEqual(len(seeds), 100)
        self.assertEqual(seeds[:3], [11042, 11043, 11044])

    def test_offline_eta_formats_seconds(self):
        """Testing: offline evaluator ETA uses stable HH:MM:SS formatting."""
        print("Testing: offline evaluator ETA formatting")

        self.assertEqual(format_seconds(0), "00:00:00")
        self.assertEqual(format_seconds(65), "00:01:05")
        self.assertEqual(format_seconds(3661), "01:01:01")

    def test_offline_evaluated_checkpoints_reads_existing_summary(self):
        """Testing: offline evaluator skips checkpoints already summarized."""
        print("Testing: offline evaluator reads existing checkpoint summaries")
        with tempfile.TemporaryDirectory() as tmpdir:
            summary_csv = Path(tmpdir) / "summary.csv"
            summary_csv.write_text(
                "checkpoint_path,controller\n"
                "runs/a/checkpoints/ppo_step_100.zip,learned_agent\n"
                "runs/a/checkpoints/ppo_step_100.zip,orca\n"
                "runs/a/checkpoints/ppo_step_200.zip,orca\n"
            )

            self.assertEqual(evaluated_checkpoints(summary_csv), {"runs/a/checkpoints/ppo_step_100.zip"})

    def test_offline_summary_row_includes_rates_and_means(self):
        """Testing: offline evaluator builds aggregate checkpoint rows."""
        print("Testing: offline evaluator summary rows")
        rows = [
            {"SUCCESS": True, "COLLISION_HUMAN": False, "SPL": 0.8, "episode_reward": 2.0},
            {"SUCCESS": False, "COLLISION_HUMAN": True, "SPL": 0.2, "episode_reward": -1.0},
        ]

        row = summary_row(Path("ppo_step_100.zip"), 100, "learned_agent", rows)

        self.assertEqual(row["checkpoint_step"], 100)
        self.assertEqual(row["success_rate"], 0.5)
        self.assertEqual(row["collision_human_rate"], 0.5)
        self.assertEqual(row["mean_spl"], 0.5)
        self.assertEqual(row["mean_episode_reward"], 0.5)

    def test_clear_testing_outputs_removes_owned_files_only(self):
        """Testing: final testing overwrite clears only configured output files."""
        print("Testing: final testing overwrite clears configured output files")
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            owned = output_dir / "test_agent_metrics.csv"
            other = output_dir / "notes.txt"
            owned.write_text("old")
            other.write_text("keep")

            clear_testing_outputs(output_dir, {})

            self.assertFalse(owned.exists())
            self.assertTrue(other.exists())

    def test_orca_robot_policy_outputs_normalized_diff_drive_action(self):
        """Testing: ORCA robot velocity is converted to the diff-drive action interface."""
        print("Testing: ORCA robot baseline action adapter")
        env = DummyORCAEnv()

        action = ORCARobotPolicy().predict(None, env=env)

        np.testing.assert_allclose(action, np.array([0.5, 0.0], dtype=np.float32))

    def test_orca_velocity_conversion_clips_to_action_bounds(self):
        """Testing: ORCA baseline action conversion remains inside action bounds."""
        print("Testing: ORCA action conversion clips to [-1, 1]")
        env = DummyORCAEnv()
        robot = DummyRobot()

        action = orca_velocity_to_action(env, robot, np.array([2.0, 2.0], dtype=np.float32), (2,))

        self.assertEqual(action.shape, (2,))
        self.assertLessEqual(float(np.max(action)), 1.0)
        self.assertGreaterEqual(float(np.min(action)), -1.0)

    def test_orca_velocity_conversion_matches_raw_socnavgym_angle_delta(self):
        """Testing: ORCA angular action mirrors SocNavGym's raw angle difference."""
        print("Testing: ORCA action conversion uses SocNavGym raw angle difference")
        env = DummyORCAEnv()
        robot = DummyRobot()
        robot.orientation = 3.0

        action = orca_velocity_to_action(env, robot, np.array([-1.0, 0.0], dtype=np.float32), (2,))

        self.assertAlmostEqual(float(action[1]), 0.14159265, places=5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
