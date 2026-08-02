from collections import deque
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import gym
import numpy as np
import socnavgym  # noqa: F401 - registers SocNavGym-v1 with gym
import torch
import yaml
from gym import spaces

from training_pipeline.architecture_extractor import ArchitectureFeaturesExtractor
from training_pipeline.observation_history_wrapper import ObservationHistoryWrapper
from training_pipeline.utils import load_yaml


ROOT = Path(__file__).resolve().parents[1]
HUMANS_CONFIG_PATH = ROOT / "env_configs" / "env_humans.yaml"
ENTITY_FEATURE_DIM = 14
HISTORY_LENGTH = 4


def _assert_interactions_disabled(test_case, config):
    env = config["env"]
    count_keys = (
        "max_h_h_dynamic_interactions",
        "max_h_h_dynamic_interactions_non_dispersing",
        "max_h_h_static_interactions",
        "max_h_h_static_interactions_non_dispersing",
        "max_h_l_interactions",
        "max_h_l_interactions_non_dispersing",
    )
    probability_keys = (
        "crowd_dispersal_probability",
        "human_laptop_dispersal_probability",
        "crowd_formation_probability",
        "human_laptop_formation_probability",
    )
    for key in count_keys + probability_keys:
        test_case.assertEqual(env[key], 0, f"{key} must remain disabled for identity-by-row history.")


def _padded_human_rows(records, capacity):
    rows = np.zeros((capacity, ENTITY_FEATURE_DIM), dtype=np.float32)
    for index, (_, values) in enumerate(records):
        rows[index] = values
    return rows


class TestObservationHistoryIdentity(unittest.TestCase):
    """Verify that human identities retain one temporal row from SocNavGym to the BiGRU."""

    def test_socnavgym_human_ids_remain_in_history_slots(self):
        """SocNavGym row order remains stable and the history wrapper preserves it."""
        print("Testing: SocNavGym human IDs remain in the same observation-history slots")
        config = load_yaml(str(HUMANS_CONFIG_PATH))
        _assert_interactions_disabled(self, config)
        # Keep the integration test complete but short. All identity-relevant
        # entity and interaction settings remain identical to the real config.
        config["episode"]["episode_length"] = 24

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "identity_test_socnav.yaml"
            config_path.write_text(yaml.safe_dump(config, sort_keys=False))
            base_wrapper = gym.make("SocNavGym-v1", config=str(config_path))
            base_env = base_wrapper.unwrapped
            env = ObservationHistoryWrapper(
                base_wrapper,
                history_length=HISTORY_LENGTH,
                entity_keys=("humans",),
            )

            original_get_entity_obs = base_env._get_entity_obs
            records = []

            def record_entity_observation(entity):
                values = np.asarray(original_get_entity_obs(entity), dtype=np.float32).reshape(-1)
                if entity.name == "human":
                    records.append((entity.id, values.copy()))
                return values

            try:
                with patch.object(base_env, "_get_entity_obs", side_effect=record_entity_observation):
                    for seed in (1201, 1202):
                        records.clear()
                        observation, _ = env.reset(seed=seed)
                        initial_ids = tuple(entity.id for entity in base_env.static_humans + base_env.dynamic_humans)
                        observed_ids = tuple(entity_id for entity_id, _ in records)
                        self.assertEqual(observed_ids, initial_ids)

                        capacity = observation["humans"].shape[0]
                        current_rows = _padded_human_rows(records, capacity)
                        expected_history = deque(
                            (current_rows.copy() for _ in range(HISTORY_LENGTH)),
                            maxlen=HISTORY_LENGTH,
                        )
                        np.testing.assert_allclose(
                            observation["humans"],
                            np.stack(expected_history, axis=1),
                        )

                        completed = False
                        while not completed:
                            records.clear()
                            observation, _, terminated, truncated, _ = env.step(
                                np.zeros(3, dtype=np.float32)
                            )
                            observed_ids = tuple(entity_id for entity_id, _ in records)
                            self.assertEqual(observed_ids, initial_ids)

                            current_rows = _padded_human_rows(records, capacity)
                            expected_history.append(current_rows)
                            np.testing.assert_allclose(
                                observation["humans"],
                                np.stack(expected_history, axis=1),
                            )
                            completed = bool(terminated or truncated)
            finally:
                env.close()

    def test_extractor_passes_human_history_to_bigru_without_reordering(self):
        """The extractor's human BiGRU receives the wrapper's exact entity/time ordering."""
        print("Testing: feature extractor passes human history to the BiGRU without reordering")
        num_humans = 5
        observation_space = spaces.Dict(
            {
                "robot": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(HISTORY_LENGTH, 9),
                    dtype=np.float32,
                ),
                "humans": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(num_humans, HISTORY_LENGTH, ENTITY_FEATURE_DIM),
                    dtype=np.float32,
                ),
            }
        )
        extractor = ArchitectureFeaturesExtractor(
            observation_space,
            architecture_name="social_context_fusion",
            architecture_config_path=str(ROOT / "architectures" / "social_context_fusion" / "config.yaml"),
            entity_keys=("humans",),
            entity_feature_dim=ENTITY_FEATURE_DIM,
        )

        human_history = torch.arange(
            num_humans * HISTORY_LENGTH * ENTITY_FEATURE_DIM,
            dtype=torch.float32,
        ).reshape(1, num_humans, HISTORY_LENGTH, ENTITY_FEATURE_DIM)
        human_history[:, :, :, :6] = 0.0
        human_history[:, :, :, 1] = 1.0
        captured = []

        def capture_bigru_input(_module, inputs):
            captured.append(inputs[0].detach().clone())

        handle = extractor.architecture.entity_encoder.gru.register_forward_pre_hook(capture_bigru_input)
        try:
            with torch.no_grad():
                extractor(
                    {
                        "robot": torch.zeros((1, HISTORY_LENGTH, 9), dtype=torch.float32),
                        "humans": human_history,
                    }
                )
        finally:
            handle.remove()

        self.assertEqual(len(captured), 1)
        expected = human_history.reshape(num_humans, HISTORY_LENGTH, ENTITY_FEATURE_DIM)
        torch.testing.assert_close(captured[0], expected)


if __name__ == "__main__":
    unittest.main()
