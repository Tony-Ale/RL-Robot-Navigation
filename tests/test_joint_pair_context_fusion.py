from copy import deepcopy
from pathlib import Path
import unittest

import torch
from torch import nn

from architectures.joint_pair_context_fusion import (
    JointPairContextFusionNet,
    load_architecture_config,
)
from training_pipeline.architecture_extractor import architecture_feature_dim


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "architectures" / "joint_pair_context_fusion" / "config.yaml"


class TestJointPairContextFusion(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.config = load_architecture_config(str(CONFIG_PATH))

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

    def test_encoded_robot_context_preserves_128_feature_contract(self):
        model = JointPairContextFusionNet(self.config).eval()
        robot, entities, mask = self.observations()

        result = model(robot, entities, mask, return_attention=True)

        self.assertFalse(any(isinstance(module, nn.GRU) for module in model.modules()))
        self.assertEqual(tuple(result["robot_context"].shape), (2, 64))
        self.assertEqual(tuple(result["social_context"].shape), (2, 64))
        self.assertEqual(tuple(model(robot, entities, mask, False).shape), (2, 128))
        self.assertEqual(architecture_feature_dim("joint_pair_context_fusion", self.config), 128)

    def test_disabled_robot_encoding_passes_current_robot_features_unchanged(self):
        config = deepcopy(self.config)
        config["robot_encoding"]["enabled"] = False
        model = JointPairContextFusionNet(config).eval()
        robot, entities, mask = self.observations()

        result = model(robot, entities, mask, return_attention=True)

        torch.testing.assert_close(result["robot_context"], robot)
        self.assertEqual(tuple(model(robot, entities, mask, False).shape), (2, 73))
        self.assertEqual(
            architecture_feature_dim("joint_pair_context_fusion", config),
            73,
        )
        self.assertEqual(
            architecture_feature_dim("joint_pair_context_fusion", config, effective_robot_dim=17),
            81,
        )

    def test_joint_mlp_receives_robot_entity_pairs_and_masking_is_truthful(self):
        model = JointPairContextFusionNet(self.config).eval()
        robot, entities, mask = self.observations()

        result = model(robot, entities, mask, return_attention=True)

        first_linear = next(module for module in model.joint_mlp if isinstance(module, nn.Linear))
        self.assertEqual(first_linear.in_features, 23)
        self.assertEqual(tuple(result["interaction_embedding"].shape), (2, 3, 100))
        self.assertTrue(torch.all(result["attention_weights"][:, 2] == 0))
        torch.testing.assert_close(
            result["attention_weights"].sum(dim=1),
            torch.ones(2),
        )

    def test_history_input_uses_only_current_robot_and_entity_observations(self):
        model = JointPairContextFusionNet(self.config).eval()
        robot, entities, mask = self.observations()
        robot_history = robot.unsqueeze(1).repeat(1, 4, 1)
        entity_history = entities.unsqueeze(2).repeat(1, 1, 4, 1)
        robot_history[:, :-1] += 100.0
        entity_history[:, :, :-1] += 100.0

        current = model(robot, entities, mask, return_attention=False)
        history = model(robot_history, entity_history, mask, return_attention=False)

        torch.testing.assert_close(current, history)

    def test_entity_order_does_not_change_the_pooled_context(self):
        model = JointPairContextFusionNet(self.config).eval()
        robot, entities, mask = self.observations()
        baseline = model(robot, entities, mask, return_attention=True)
        permutation = torch.tensor([1, 0, 2])
        permuted = model(
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


if __name__ == "__main__":
    unittest.main()
