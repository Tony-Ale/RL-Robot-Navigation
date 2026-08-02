import unittest
from pathlib import Path

from environment_inspection.episode_analysis import collect_analysis_episodes
from environment_inspection.output_utils import outcome_from_info
from testing_pipeline.checkpoints import checkpoint_step
from training_pipeline.episode_runtime import is_planner_reset_failure, reset_env, reset_policy
from training_pipeline.training_runtime import learn_model, managed_training_environment


class _Policy:
    controller_name = "test"

    def __init__(self):
        self.reset_count = 0

    def reset(self):
        self.reset_count += 1

    def predict(self, observation, env):
        return 0


class _AnalysisEnv:
    def __init__(self):
        self.reset_seeds = []
        self.step_count = 0
        self.action_space = self

    def reset(self, seed=None):
        self.reset_seeds.append(seed)
        if seed == 10:
            raise RuntimeError("Planner produced no waypoints after reset")
        self.step_count = 0
        return {"seed": seed}, {}

    def step(self, action):
        self.step_count += 1
        done = self.step_count == 2
        return {}, 0.0, done, False, {"SUCCESS": done}

    def sample(self):
        return 0


class _ClosableEnv:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class _Model:
    def __init__(self):
        self.kwargs = None

    def learn(self, **kwargs):
        self.kwargs = kwargs


class SharedRuntimeTests(unittest.TestCase):
    def test_episode_runtime_normalizes_legacy_reset_and_resets_supported_policy(self):
        class LegacyEnv:
            def reset(self, seed=None):
                return {"seed": seed}

        policy = _Policy()
        observation, info = reset_env(LegacyEnv(), 7)
        reset_policy(policy)
        reset_policy(None)

        self.assertEqual(observation, {"seed": 7})
        self.assertEqual(info, {})
        self.assertEqual(policy.reset_count, 1)

    def test_planner_failure_detection_has_one_message_contract(self):
        self.assertTrue(is_planner_reset_failure(RuntimeError("Planner produced no waypoints after reset")))
        self.assertFalse(is_planner_reset_failure(RuntimeError("different failure")))

    def test_analysis_collection_skips_invalid_seeds_and_resets_memory_once_per_valid_episode(self):
        env = _AnalysisEnv()
        policy = _Policy()
        collection = collect_analysis_episodes(
            env,
            policy,
            base_seed=10,
            episode_count=2,
            max_steps=5,
            start_trace=lambda active_env: [active_env.step_count],
            update_trace=lambda trace, active_env: trace.append(active_env.step_count),
        )

        self.assertEqual(env.reset_seeds, [10, 11, 12])
        self.assertEqual(collection.skipped_resets, 1)
        self.assertEqual([episode.seed for episode in collection.episodes], [11, 12])
        self.assertEqual([episode.steps for episode in collection.episodes], [2, 2])
        self.assertEqual([episode.trace for episode in collection.episodes], [[0, 1, 2], [0, 1, 2]])
        self.assertEqual(policy.reset_count, 2)

    def test_checkpoint_and_outcome_helpers_define_shared_formats(self):
        self.assertEqual(checkpoint_step(Path("ppo_final_step_125000.zip")), 125000)
        self.assertEqual(checkpoint_step(Path("checkpoint.zip")), -1)
        self.assertEqual(outcome_from_info({"SUCCESS": True}), "success")
        self.assertEqual(outcome_from_info({}, fallback="max_steps_reached"), "max_steps_reached")

    def test_training_environment_closes_when_setup_fails(self):
        env = _ClosableEnv()
        with self.assertRaisesRegex(RuntimeError, "setup failed"):
            with managed_training_environment(env):
                raise RuntimeError("setup failed")
        self.assertTrue(env.closed)

    def test_shared_learning_call_passes_pipeline_arguments(self):
        model = _Model()
        config = {
            "experiment": {"name": "test-run"},
            "training": {
                "total_timesteps": 100,
                "log_interval": 4,
                "reset_num_timesteps": False,
            },
        }

        learn_model(config, model, callbacks="callbacks")

        self.assertEqual(
            model.kwargs,
            {
                "total_timesteps": 100,
                "callback": "callbacks",
                "log_interval": 4,
                "reset_num_timesteps": False,
                "tb_log_name": "test-run",
            },
        )


if __name__ == "__main__":
    unittest.main()
