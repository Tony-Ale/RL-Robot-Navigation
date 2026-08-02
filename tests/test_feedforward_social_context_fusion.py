from pathlib import Path
import unittest

import torch
from torch import nn

from architectures.feedforward_social_context_fusion import (
    FeedForwardSocialContextFusionNet,
    load_architecture_config,
)
from training_pipeline.utils import load_yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "architectures" / "feedforward_social_context_fusion" / "config.yaml"
SOCIAL_CONFIG_PATH = ROOT / "architectures" / "social_context_fusion" / "config.yaml"


class TestFeedForwardSocialContextFusion(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(13)
        self.model = FeedForwardSocialContextFusionNet.from_yaml(str(CONFIG_PATH)).eval()

    @staticmethod
    def observations():
        robot = torch.randn(2, 9)
        entities = torch.zeros(2, 3, 14)
        entities[:, 0, 1] = 1.0
        entities[:, 0, 6:] = torch.randn(2, 8)
        entities[:, 1, 1] = 1.0
        entities[:, 1, 6:] = torch.randn(2, 8)
        mask = torch.tensor([[True, True, False], [True, True, False]])
        return robot, entities, mask

    def test_uses_only_mlp_encoders_and_preserves_social_context_contract(self):
        robot, entities, mask = self.observations()

        result = self.model(robot, entities, mask, return_attention=True)

        self.assertFalse(any(isinstance(module, nn.GRU) for module in self.model.modules()))
        self.assertEqual(tuple(result["robot_context"].shape), (2, 64))
        self.assertEqual(tuple(result["social_context"].shape), (2, 64))
        self.assertEqual(tuple(result["attention_weights"].shape), (2, 3))
        self.assertEqual(tuple(self.model(robot, entities, mask, False).shape), (2, 128))
        self.assertTrue(torch.all(result["attention_weights"][:, 2] == 0))
        self.assertTrue(torch.allclose(result["attention_weights"].sum(dim=1), torch.ones(2)))

    def test_history_shaped_inputs_use_only_the_latest_observation(self):
        robot, entities, mask = self.observations()
        robot_history = robot.unsqueeze(1).repeat(1, 4, 1)
        entity_history = entities.unsqueeze(2).repeat(1, 1, 4, 1)
        changed_past_robot = robot_history.clone()
        changed_past_entities = entity_history.clone()
        changed_past_robot[:, :-1] += 100.0
        changed_past_entities[:, :, :-1] += 100.0

        current = self.model(robot, entities, mask, return_attention=False)
        history = self.model(
            changed_past_robot, changed_past_entities, mask, return_attention=False
        )

        torch.testing.assert_close(current, history)

    def test_entity_mlp_is_shared_and_fusion_is_permutation_invariant(self):
        robot, entities, mask = self.observations()
        baseline = self.model(robot, entities, mask, return_attention=True)
        permutation = torch.tensor([1, 0, 2])
        permuted = self.model(
            robot,
            entities[:, permutation],
            mask[:, permutation],
            return_attention=True,
        )

        torch.testing.assert_close(baseline["social_context"], permuted["social_context"])
        torch.testing.assert_close(
            baseline["attention_weights"][:, permutation],
            permuted["attention_weights"],
        )

    def test_all_padded_entities_produce_zero_finite_social_context(self):
        robot = torch.randn(2, 9)
        entities = torch.zeros(2, 3, 14)
        mask = torch.zeros(2, 3, dtype=torch.bool)

        result = self.model(robot, entities, mask, return_attention=True)

        self.assertTrue(torch.all(result["attention_weights"] == 0))
        self.assertTrue(torch.all(result["social_context"] == 0))
        self.assertTrue(torch.isfinite(result["robot_context"]).all())

    def test_robot_and_entity_encoders_receive_gradients(self):
        self.model.train()
        robot, entities, mask = self.observations()

        self.model(robot, entities, mask, return_attention=False).sum().backward()

        robot_gradients = [parameter.grad for parameter in self.model.robot_encoder.parameters()]
        entity_gradients = [parameter.grad for parameter in self.model.entity_encoder.parameters()]
        self.assertTrue(any(gradient is not None and torch.any(gradient != 0) for gradient in robot_gradients))
        self.assertTrue(any(gradient is not None and torch.any(gradient != 0) for gradient in entity_gradients))

    def test_fusion_settings_match_social_context_fusion(self):
        feedforward = load_architecture_config(str(CONFIG_PATH))
        social = load_yaml(str(SOCIAL_CONFIG_PATH))

        for section in (
            "interaction_embedding",
            "feature_reduction",
            "attention",
            "prediction_head",
        ):
            self.assertEqual(feedforward[section], social[section])

        self.assertEqual(feedforward["observation_encoding"]["robot_output_dim"], 64)
        self.assertEqual(feedforward["observation_encoding"]["entity_output_dim"], 64)


if __name__ == "__main__":
    unittest.main()
