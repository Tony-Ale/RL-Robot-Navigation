from pathlib import Path
import unittest

import torch

from architectures.crowd_context_fusion import CrowdContextFusionNet


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "architectures" / "crowd_context_fusion" / "config.yaml"


class TestCrowdContextFusionNet(unittest.TestCase):
    """Tests for the robot-MLP plus crowd-BiGRU architecture."""

    def _model(self):
        print("Testing setup: loading CrowdContextFusionNet from YAML")
        torch.manual_seed(7)
        return CrowdContextFusionNet.from_yaml(str(CONFIG_PATH))

    def test_forward_returns_expected_output_and_attention_shapes(self):
        """Testing: model returns value output and one attention weight per entity."""
        print("Testing: CrowdContextFusionNet output and attention shapes")
        model = self._model()

        robot = torch.randn(2, 9)
        entities = torch.randn(2, 5, 4, 14)
        mask = torch.ones(2, 5, dtype=torch.bool)

        result = model(robot, entities, mask)

        self.assertIsNone(result["output"])
        self.assertEqual(result["attention_weights"].shape, (2, 5))
        self.assertEqual(result["robot_context"].shape, (2, model.config["robot_embedding"]["output_dim"]))
        self.assertEqual(result["crowd_context"].shape, (2, 5, model.crowd_encoder.output_dim))
        self.assertEqual(result["social_context"].shape, (2, model.config["feature_reduction"]["output_dim"]))

    def test_attention_mask_sets_padded_entity_weights_to_zero(self):
        """Testing: padded entities receive zero attention weight."""
        print("Testing: CrowdContextFusionNet attention mask behavior")
        model = self._model()

        robot = torch.randn(2, 9)
        entities = torch.randn(2, 4, 3, 14)
        mask = torch.tensor([[True, True, False, False], [True, False, True, False]])

        result = model(robot, entities, mask)

        self.assertTrue(torch.allclose(result["attention_weights"][~mask], torch.zeros_like(result["attention_weights"][~mask])))
        self.assertTrue(torch.allclose(result["attention_weights"].sum(dim=1), torch.ones(2), atol=1e-6))

    def test_zero_entities_returns_empty_attention_and_valid_output(self):
        """Testing: model can run when no humans/entities are available."""
        print("Testing: CrowdContextFusionNet zero-entity path")
        model = self._model()

        robot = torch.randn(3, 9)
        entities = torch.randn(3, 0, 5, 14)

        result = model(robot, entities)

        self.assertIsNone(result["output"])
        self.assertEqual(result["attention_weights"].shape, (3, 0))
        self.assertEqual(result["crowd_context"].shape, (3, 0, model.crowd_encoder.output_dim))

    def test_robot_history_input_uses_latest_robot_observation(self):
        """Testing: robot history input is accepted and reduced to the latest observation."""
        print("Testing: CrowdContextFusionNet accepts robot history input")
        model = self._model()

        robot_history = torch.randn(2, 6, 9)
        entities = torch.randn(2, 3, 4, 14)

        result = model(robot_history, entities)

        self.assertIsNone(result["output"])
        self.assertEqual(result["robot_context"].shape, (2, model.config["robot_embedding"]["output_dim"]))

    def test_forward_without_attention_returns_fused_features_when_head_disabled(self):
        """Testing: disabled prediction head exposes fused PPO features."""
        print("Testing: CrowdContextFusionNet returns fused features without attention")
        model = self._model()

        robot = torch.randn(2, 9)
        entities = torch.randn(2, 3, 4, 14)

        features = model(robot, entities, return_attention=False)

        expected_dim = model.config["robot_embedding"]["output_dim"] + model.config["feature_reduction"]["output_dim"]
        self.assertEqual(features.shape, (2, expected_dim))


if __name__ == "__main__":
    unittest.main()
