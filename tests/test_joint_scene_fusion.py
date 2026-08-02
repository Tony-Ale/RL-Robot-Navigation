import unittest

import torch

from architectures.joint_scene_fusion import JointSceneFusionNet, load_architecture_config


class TestJointSceneFusionNet(unittest.TestCase):
    """Tests for the single-BiGRU joint robot/entity scene fusion architecture."""

    def make_model(self):
        """Create the model from YAML so the test matches the configured architecture."""
        print("Testing setup: loading JointSceneFusionNet from YAML")
        config = load_architecture_config("architectures/joint_scene_fusion/config.yaml")
        return JointSceneFusionNet(config)

    def test_forward_returns_expected_output_and_attention_shapes(self):
        """Testing: model returns value output and one attention weight per entity."""
        print("Testing: JointSceneFusionNet output and attention shapes")
        model = self.make_model()
        robot = torch.randn(4, 9)
        entities = torch.randn(4, 6, 1, 14)
        mask = torch.ones(4, 6, dtype=torch.bool)

        result = model(robot, entities, mask)

        self.assertIsNone(result["output"])
        self.assertEqual(result["attention_weights"].shape, (4, 6))
        self.assertEqual(result["robot_context"].shape, (4, model.config["robot_projection"]["output_dim"]))
        self.assertEqual(result["scene_context"].shape, (4, 6, model.scene_encoder.output_dim))
        self.assertEqual(result["social_context"].shape, (4, model.config["feature_reduction"]["output_dim"]))

    def test_attention_mask_sets_padded_entity_weights_to_zero(self):
        """Testing: padded entities receive zero attention weight."""
        print("Testing: JointSceneFusionNet attention mask behavior")
        model = self.make_model()
        robot = torch.randn(2, 9)
        entities = torch.randn(2, 4, 1, 14)
        mask = torch.tensor([[True, True, False, False], [True, False, True, False]])

        result = model(robot, entities, mask)

        self.assertTrue(torch.allclose(result["attention_weights"][~mask], torch.zeros_like(result["attention_weights"][~mask])))
        self.assertTrue(torch.allclose(result["attention_weights"].sum(dim=1), torch.ones(2), atol=1e-6))

    def test_zero_entities_returns_empty_attention_and_valid_output(self):
        """Testing: model can run when no entities are available."""
        print("Testing: JointSceneFusionNet zero-entity path")
        model = self.make_model()
        robot = torch.randn(3, 9)
        entities = torch.empty(3, 0, 1, 14)

        result = model(robot, entities)

        self.assertIsNone(result["output"])
        self.assertEqual(result["attention_weights"].shape, (3, 0))
        self.assertEqual(result["scene_context"].shape, (3, 0, model.scene_encoder.output_dim))
        self.assertEqual(result["social_context"].shape, (3, model.config["feature_reduction"]["output_dim"]))

    def test_robot_and_entity_histories_are_paired_by_timestep(self):
        """Testing: each entity frame is paired with the matching robot frame."""
        print("Testing: JointSceneFusionNet aligns robot and entity history timesteps")
        model = self.make_model()
        robot_history = torch.arange(27, dtype=torch.float32).reshape(1, 3, 9)
        entities = torch.arange(84, dtype=torch.float32).reshape(1, 2, 3, 14)
        captured = []
        hook = model.scene_encoder.gru.register_forward_pre_hook(lambda _module, inputs: captured.append(inputs[0].detach().clone()))

        try:
            model(robot_history, entities)
        finally:
            hook.remove()

        expected_robot = robot_history[:, None, :, :].expand(-1, 2, -1, -1)
        expected_pairs = torch.cat([expected_robot, entities], dim=-1).reshape(2, 3, 23)
        torch.testing.assert_close(captured[0], expected_pairs)

    def test_mismatched_history_lengths_raise_error(self):
        """Testing: robot and entity frames cannot be paired when lengths differ."""
        print("Testing: JointSceneFusionNet rejects mismatched history lengths")
        model = self.make_model()

        with self.assertRaisesRegex(ValueError, "same number of timesteps"):
            model(torch.randn(2, 5, 9), torch.randn(2, 3, 4, 14))

    def test_forward_without_attention_returns_fused_features_when_head_disabled(self):
        """Testing: disabled prediction head exposes fused PPO features."""
        print("Testing: JointSceneFusionNet returns fused features without attention")
        model = self.make_model()
        robot = torch.randn(2, 9)
        entities = torch.randn(2, 3, 1, 14)

        features = model(robot, entities, return_attention=False)

        expected_dim = model.config["robot_projection"]["output_dim"] + model.config["feature_reduction"]["output_dim"]
        self.assertEqual(features.shape, (2, expected_dim))


if __name__ == "__main__":
    unittest.main(verbosity=2)
