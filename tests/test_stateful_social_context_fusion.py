import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from architectures.stateful_social_context_fusion import StatefulSocialContextFusionNet, load_architecture_config
from stateful_training_pipeline.policy import StatefulSocialContextPolicy
from stateful_training_pipeline.policies import StatefulLearnedAgentPolicy
from stateful_training_pipeline.recurrent_ppo import StatefulSocialRecurrentPPO
from stateful_training_pipeline.train import validate_config
from training_pipeline.utils import load_yaml


ARCHITECTURE_CONFIG = "architectures/stateful_social_context_fusion/config.yaml"


class _TinySocialEnv(gym.Env):
    def __init__(self):
        self.observation_space = spaces.Dict(
            {
                "robot": spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32),
                "humans": spaces.Box(-1.0, 1.0, shape=(2, 14), dtype=np.float32),
            }
        )
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.steps = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        return self._observation(), {}

    def step(self, action):
        self.steps += 1
        terminated = self.steps >= 4
        return self._observation(), 1.0, terminated, False, {}

    def _observation(self):
        robot = np.array([self.steps / 4.0, 0.1, -0.1], dtype=np.float32)
        humans = np.zeros((2, 14), dtype=np.float32)
        humans[0, :3] = (1.0, self.steps / 4.0, 0.2)
        return {"robot": robot, "humans": humans}


class _TinyMixedEntityEnv(_TinySocialEnv):
    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Dict(
            {
                "robot": spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32),
                "humans": spaces.Box(-1.0, 1.0, shape=(2, 14), dtype=np.float32),
                "tables": spaces.Box(-1.0, 1.0, shape=(1, 14), dtype=np.float32),
                "walls": spaces.Box(-1.0, 1.0, shape=(4, 14), dtype=np.float32),
            }
        )

    def _observation(self):
        observation = super()._observation()
        tables = np.zeros((1, 14), dtype=np.float32)
        tables[0, :2] = (1.0, 0.5)
        walls = np.zeros((4, 14), dtype=np.float32)
        walls[0, :2] = (1.0, -0.5)
        observation.update(tables=tables, walls=walls)
        return observation


class StatefulSocialContextFusionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        config = load_architecture_config(ARCHITECTURE_CONFIG)
        self.network = StatefulSocialContextFusionNet(config, robot_input_dim=3, num_entities=2)
        self.network.eval()

    def test_state_is_deterministic_and_episode_start_resets_it(self):
        robot = torch.tensor([[0.1, 0.2, 0.3]])
        humans = torch.zeros((1, 2, 14))
        humans[0, 0, :2] = torch.tensor([0.4, 0.5])
        mask = torch.tensor([[True, False]])
        initial = torch.zeros((1, 1, self.network.state_size))

        first_features, first_state = self.network.forward_sequence(
            robot, humans, mask, initial, torch.ones(1)
        )
        repeated_features, repeated_state = self.network.forward_sequence(
            robot, humans, mask, initial, torch.ones(1)
        )
        continued_features, _ = self.network.forward_sequence(
            robot, humans, mask, first_state, torch.zeros(1)
        )
        reset_features, reset_state = self.network.forward_sequence(
            robot, humans, mask, first_state, torch.ones(1)
        )

        torch.testing.assert_close(first_features, repeated_features)
        torch.testing.assert_close(first_state, repeated_state)
        torch.testing.assert_close(first_features, reset_features)
        torch.testing.assert_close(first_state, reset_state)
        self.assertFalse(torch.allclose(first_features, continued_features))

    def test_padded_human_state_remains_zero(self):
        robot = torch.ones((1, 3))
        humans = torch.zeros((1, 2, 14))
        humans[0, 0, 0] = 1.0
        mask = torch.tensor([[True, False]])
        initial = torch.zeros((1, 1, self.network.state_size))

        _, packed_state = self.network.forward_sequence(robot, humans, mask, initial, torch.ones(1))
        _, human_state = self.network._unpack_state(packed_state)

        self.assertGreater(torch.linalg.vector_norm(human_state[:, :, 0]).item(), 0.0)
        torch.testing.assert_close(human_state[:, :, 1], torch.zeros_like(human_state[:, :, 1]))
        self.assertIsNotNone(self.network.last_attention_weights)
        self.assertEqual(self.network.last_attention_weights[0, 1].item(), 0.0)

    def test_recurrent_ppo_learns_and_round_trips_checkpoint(self):
        env = _TinySocialEnv()
        policy_kwargs = {
            "architecture_config_path": ARCHITECTURE_CONFIG,
            "entity_keys": ["humans"],
            "entity_feature_dim": 14,
            "mask_zero_entities": True,
            "include_waypoint_features": False,
            "net_arch": [],
        }
        model = StatefulSocialRecurrentPPO(
            StatefulSocialContextPolicy,
            env,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            policy_kwargs=policy_kwargs,
            seed=11,
            verbose=0,
        )
        model.learn(total_timesteps=8)
        observation, _ = env.reset(seed=13)
        action, state = model.predict(
            observation,
            state=None,
            episode_start=np.ones((1,), dtype=bool),
            deterministic=True,
        )

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "stateful_model"
            model.save(checkpoint)
            loaded = StatefulSocialRecurrentPPO.load(checkpoint, env=env)
            loaded_action, loaded_state = loaded.predict(
                observation,
                state=None,
                episode_start=np.ones((1,), dtype=bool),
                deterministic=True,
            )
            policy_checkpoint = Path(directory) / "stateful_policy"
            model.policy.save(policy_checkpoint)
            loaded_policy = StatefulSocialContextPolicy.load(policy_checkpoint)
            policy_action, policy_state = loaded_policy.predict(
                observation,
                state=None,
                episode_start=np.ones((1,), dtype=bool),
                deterministic=True,
            )

        np.testing.assert_allclose(action, loaded_action)
        np.testing.assert_allclose(state[0], loaded_state[0])
        np.testing.assert_allclose(state[1], loaded_state[1])
        np.testing.assert_allclose(action, policy_action)
        np.testing.assert_allclose(state[0], policy_state[0])

    def test_policy_construction_does_not_allocate_lstms(self):
        observation_space = _TinySocialEnv().observation_space
        calls = []
        original = torch.nn.LSTM

        def record_lstm(*args, **kwargs):
            calls.append(int(args[1] if len(args) > 1 else kwargs["hidden_size"]))
            return original(*args, **kwargs)

        with patch("sb3_contrib.common.recurrent.policies.nn.LSTM", side_effect=record_lstm):
            StatefulSocialContextPolicy(
                observation_space,
                _TinySocialEnv().action_space,
                lambda _: 0.001,
                architecture_config_path=ARCHITECTURE_CONFIG,
                entity_keys=["humans"],
                entity_feature_dim=14,
                net_arch=[],
            )
        self.assertEqual(calls, [])

    def test_entity_feature_dimensions_must_match(self):
        with self.assertRaisesRegex(ValueError, "must match"):
            StatefulSocialContextPolicy(
                _TinySocialEnv().observation_space,
                _TinySocialEnv().action_space,
                lambda _: 0.001,
                architecture_config_path=ARCHITECTURE_CONFIG,
                entity_keys=["humans"],
                entity_feature_dim=7,
                net_arch=[],
            )

    def test_policy_accepts_multiple_fixed_entity_keys(self):
        env = _TinyMixedEntityEnv()
        policy = StatefulSocialContextPolicy(
            env.observation_space,
            env.action_space,
            lambda _: 0.001,
            architecture_config_path=ARCHITECTURE_CONFIG,
            entity_keys=["humans", "tables", "walls"],
            entity_feature_dim=14,
            net_arch=[],
        )

        self.assertEqual(policy.entity_keys, ("humans", "tables", "walls"))
        self.assertEqual(policy.architecture.num_entities, 7)

    def test_original_humans_only_config_remains_valid(self):
        config = deepcopy(load_yaml("stateful_training_pipeline/config.yaml"))

        validate_config(config)

    def test_mixed_native_entities_are_valid(self):
        config = deepcopy(load_yaml("stateful_training_pipeline/config.yaml"))
        config["architecture"]["entity_keys"] = ["humans", "laptops", "tables", "plants"]

        validate_config(config)

    def test_stateful_walls_require_enabled_all_mode(self):
        config = deepcopy(load_yaml("stateful_training_pipeline/config.yaml"))
        wall_config = config["wrappers"]["nearest_wall_segments"]
        wall_key = wall_config.get("observation_key", "walls")
        config["architecture"]["entity_keys"] = ["humans", wall_key]
        wall_config["enabled"] = False
        wall_config["mode"] = "nearest"

        with self.assertRaisesRegex(ValueError, "enabled"):
            validate_config(config)

        wall_config["enabled"] = True
        with self.assertRaisesRegex(ValueError, "mode: all"):
            validate_config(config)

        wall_config["mode"] = "all"
        validate_config(config)

    def test_duplicate_and_unknown_entity_keys_are_rejected(self):
        config = deepcopy(load_yaml("stateful_training_pipeline/config.yaml"))
        config["architecture"]["entity_keys"] = ["humans", "humans"]
        with self.assertRaisesRegex(ValueError, "duplicates"):
            validate_config(config)

        config["architecture"]["entity_keys"] = ["humans", "waypoint_features"]
        with self.assertRaisesRegex(ValueError, "unsupported"):
            validate_config(config)

    def test_interaction_probabilities_are_rejected(self):
        config = deepcopy(load_yaml("stateful_training_pipeline/config.yaml"))
        environment = deepcopy(load_yaml(config["environment"]["config_path"]))
        environment["env"]["crowd_formation_probability"] = 0.1
        with patch("stateful_training_pipeline.train.load_yaml", return_value=environment):
            with self.assertRaisesRegex(ValueError, "crowd_formation_probability"):
                validate_config(config)

    def test_stateful_policy_adapter_resets_on_environment_reset(self):
        class Model:
            def __init__(self):
                self.states = []

            def predict(self, observation, state, episode_start, deterministic):
                self.states.append((state, bool(episode_start[0])))
                return np.zeros((2,), dtype=np.float32), len(self.states)

        class BaseEnv:
            ticks = 0

        class Env:
            unwrapped = BaseEnv()

        model = Model()
        policy = StatefulLearnedAgentPolicy(model)
        env = Env()
        policy.predict({}, env)
        env.unwrapped.ticks = 1
        policy.predict({}, env)
        env.unwrapped.ticks = 0
        policy.predict({}, env)

        self.assertEqual(model.states, [(None, True), (1, False), (None, True)])


if __name__ == "__main__":
    unittest.main()
