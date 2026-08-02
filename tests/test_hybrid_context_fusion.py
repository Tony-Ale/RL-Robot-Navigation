from pathlib import Path
import unittest

import torch

from architectures.hybrid_context_fusion import HybridContextFusionNet


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "architectures" / "hybrid_context_fusion" / "config.yaml"


class TestHybridContextFusion(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.model = HybridContextFusionNet.from_yaml(str(CONFIG_PATH))
        self.model.eval()

    def test_forward_preserves_social_context_contract(self):
        robot = torch.randn(2, 4, 9)
        entities = torch.zeros(2, 3, 4, 14)
        entities[:, 0, :, 1] = 1.0  # Human.
        entities[:, 1, :, 3] = 1.0  # Table.
        mask = torch.tensor([[True, True, False], [True, True, False]])

        result = self.model(robot, entities, mask, return_attention=True)

        self.assertEqual(tuple(result["robot_context"].shape), (2, 40))
        self.assertEqual(tuple(result["social_context"].shape), (2, 45))
        self.assertEqual(tuple(result["attention_weights"].shape), (2, 3))
        self.assertTrue(torch.all(result["attention_weights"][:, 2] == 0))
        self.assertTrue(torch.isfinite(result["social_context"]).all())

    def test_only_humans_use_past_observations(self):
        entities = torch.zeros(1, 2, 3, 14)
        entities[:, 0, :, 1] = 1.0  # Human.
        entities[:, 1, :, 5] = 1.0  # Wall.
        changed_history = entities.clone()
        changed_history[:, :, 0, 6:] = 3.0
        mask = torch.ones(1, 2, dtype=torch.bool)

        original = self.model._encode_entities(entities, mask)
        changed = self.model._encode_entities(changed_history, mask)

        self.assertFalse(torch.allclose(original[:, 0], changed[:, 0]))
        self.assertTrue(torch.allclose(original[:, 1], changed[:, 1]))

    def test_human_and_static_encoders_receive_gradients(self):
        self.model.train()
        robot = torch.randn(1, 3, 9)
        entities = torch.zeros(1, 2, 3, 14)
        entities[:, 0, :, 1] = 1.0
        entities[:, 0, :, 6:] = torch.randn(1, 3, 8)
        entities[:, 1, :, 3] = 1.0
        entities[:, 1, :, 6:] = torch.randn(1, 3, 8)

        output = self.model(robot, entities, torch.ones(1, 2, dtype=torch.bool), return_attention=False)
        output.sum().backward()

        human_grads = [parameter.grad for parameter in self.model.entity_encoder.parameters()]
        static_grads = [parameter.grad for parameter in self.model.static_entity_encoder.parameters()]
        self.assertTrue(any(grad is not None and torch.any(grad != 0) for grad in human_grads))
        self.assertTrue(any(grad is not None and torch.any(grad != 0) for grad in static_grads))

    def test_fusion_normalizes_human_and_static_contexts(self):
        entities = torch.zeros(1, 3, 3, 14)
        entities[:, 0, :, 1] = 1.0  # Human.
        entities[:, 0, :, 6:] = torch.randn(1, 3, 8)
        entities[:, 1, :, 3] = 1.0  # Table.
        entities[:, 1, :, 6:] = torch.randn(1, 3, 8)
        mask = torch.tensor([[True, True, False]])

        encoded = self.model._encode_entities(entities, mask)

        branch_contexts = encoded[:, :2]
        self.assertTrue(torch.allclose(branch_contexts.mean(dim=-1), torch.zeros(1, 2), atol=1e-6))
        self.assertTrue(
            torch.allclose(
                branch_contexts.var(dim=-1, unbiased=False),
                torch.ones(1, 2),
                atol=1e-3,
            )
        )
        self.assertTrue(torch.all(encoded[:, 2] == 0))


if __name__ == "__main__":
    unittest.main()
