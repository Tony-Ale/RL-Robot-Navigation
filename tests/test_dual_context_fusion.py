from copy import deepcopy
from pathlib import Path
import unittest

import torch

from architectures.dual_context_fusion import DualContextFusionNet


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "architectures" / "dual_context_fusion" / "config.yaml"


class TestDualContextFusion(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        self.model = DualContextFusionNet.from_yaml(str(CONFIG_PATH))
        self.model.eval()

    @staticmethod
    def observations():
        robot = torch.randn(2, 4, 9)
        entities = torch.zeros(2, 4, 4, 14)
        entities[:, 0, :, 1] = 1.0  # Human.
        entities[:, 0, :, 6:] = torch.randn(2, 4, 8)
        entities[:, 1, :, 5] = 1.0  # Wall.
        entities[:, 1, :, 6:] = torch.randn(2, 4, 8)
        entities[:, 2, :, 2] = 1.0  # Table.
        entities[:, 2, :, 6:] = torch.randn(2, 4, 8)
        mask = torch.tensor([[True, True, True, False], [True, True, True, False]])
        return robot, entities, mask

    def test_forward_returns_separate_fixed_size_contexts(self):
        robot, entities, mask = self.observations()

        result = self.model(robot, entities, mask, return_attention=True)

        self.assertEqual(tuple(result["robot_context"].shape), (2, 40))
        self.assertEqual(tuple(result["human_context"].shape), (2, 45))
        self.assertEqual(tuple(result["obstacle_context"].shape), (2, 45))
        self.assertEqual(tuple(result["social_context"].shape), (2, 90))
        self.assertEqual(tuple(result["human_attention_weights"].shape), (2, 4))
        self.assertEqual(tuple(result["static_attention_weights"].shape), (2, 4))
        self.assertEqual(tuple(self.model(robot, entities, mask, return_attention=False).shape), (2, 130))

    def test_attention_is_normalized_independently_with_correct_masks(self):
        robot, entities, mask = self.observations()

        result = self.model(robot, entities, mask, return_attention=True)
        human_weights = result["human_attention_weights"]
        static_weights = result["static_attention_weights"]

        self.assertTrue(torch.all(human_weights[:, 1:] == 0))
        self.assertTrue(torch.all(static_weights[:, 0] == 0))
        self.assertTrue(torch.all(static_weights[:, 3] == 0))
        self.assertTrue(torch.allclose(human_weights.sum(dim=1), torch.ones(2), atol=1e-6))
        self.assertTrue(torch.allclose(static_weights.sum(dim=1), torch.ones(2), atol=1e-6))

    def test_human_and_static_branches_are_isolated(self):
        robot, entities, mask = self.observations()
        baseline = self.model(robot, entities, mask, return_attention=True)

        changed_human = entities.clone()
        changed_human[:, 0, :, 6:] += 5.0
        human_result = self.model(robot, changed_human, mask, return_attention=True)
        self.assertFalse(torch.allclose(baseline["human_context"], human_result["human_context"]))
        self.assertTrue(torch.allclose(baseline["obstacle_context"], human_result["obstacle_context"]))

        changed_wall = entities.clone()
        changed_wall[:, 1, -1, 6:] += 5.0
        wall_result = self.model(robot, changed_wall, mask, return_attention=True)
        self.assertTrue(torch.allclose(baseline["human_context"], wall_result["human_context"]))
        self.assertFalse(torch.allclose(baseline["obstacle_context"], wall_result["obstacle_context"]))

    def test_only_humans_use_observation_history(self):
        robot, entities, mask = self.observations()
        baseline = self.model(robot, entities, mask, return_attention=True)
        changed_history = entities.clone()
        changed_history[:, 0, 0, 6:] += 4.0
        changed_history[:, 1:, 0, 6:] += 4.0

        result = self.model(robot, changed_history, mask, return_attention=True)

        self.assertFalse(torch.allclose(baseline["human_context"], result["human_context"]))
        self.assertTrue(torch.allclose(baseline["obstacle_context"], result["obstacle_context"]))

    def test_absent_branch_produces_zero_context(self):
        robot = torch.randn(2, 3, 9)
        entities = torch.zeros(2, 2, 3, 14)
        entities[:, 0, :, 1] = 1.0
        mask = torch.tensor([[True, False], [True, False]])

        result = self.model(robot, entities, mask, return_attention=True)

        self.assertTrue(torch.all(result["obstacle_context"] == 0))
        self.assertTrue(torch.all(result["static_attention_weights"] == 0))
        self.assertTrue(torch.isfinite(result["social_context"]).all())

    def test_both_entity_encoders_receive_gradients(self):
        self.model.train()
        robot, entities, mask = self.observations()

        self.model(robot, entities, mask, return_attention=False).sum().backward()

        human_grads = [parameter.grad for parameter in self.model.human_encoder.parameters()]
        static_grads = [parameter.grad for parameter in self.model.static_entity_encoder.parameters()]
        self.assertTrue(any(gradient is not None and torch.any(gradient != 0) for gradient in human_grads))
        self.assertTrue(any(gradient is not None and torch.any(gradient != 0) for gradient in static_grads))

    def test_branch_specific_dimensions_can_be_tuned_independently(self):
        config = deepcopy(self.model.config)
        config["static_entity_embedding"]["output_dim"] = 30
        config["human_interaction_embedding"]["hidden_dims"] = [80, 70]
        config["static_interaction_embedding"]["hidden_dims"] = [60, 50]
        config["human_feature_reduction"]["output_dim"] = 35
        config["static_feature_reduction"]["output_dim"] = 25
        config["human_attention"]["hidden_dims"] = [64]
        config["static_attention"]["hidden_dims"] = [32, 16]
        model = DualContextFusionNet(config).eval()
        robot, entities, mask = self.observations()

        result = model(robot, entities, mask, return_attention=True)

        self.assertEqual(tuple(result["human_context"].shape), (2, 35))
        self.assertEqual(tuple(result["obstacle_context"].shape), (2, 25))
        self.assertEqual(tuple(result["social_context"].shape), (2, 60))
        self.assertEqual(tuple(model(robot, entities, mask, return_attention=False).shape), (2, 100))


if __name__ == "__main__":
    unittest.main()
