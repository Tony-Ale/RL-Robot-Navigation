from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import torch
from torch import nn


def load_architecture_config(path: str) -> Dict:
    """Load a joint-pair context-fusion architecture config."""
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required to load architecture configs.") from exc

    with open(Path(path), "r") as file:
        return yaml.safe_load(file) or {}


def _activation(name: str) -> nn.Module:
    activations = {
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "gelu": nn.GELU,
        "elu": nn.ELU,
    }
    key = name.lower()
    if key not in activations:
        raise ValueError(f"Unsupported activation '{name}'.")
    return activations[key]()


def _build_mlp(
    input_dim: int,
    hidden_dims: Iterable[int],
    output_dim: int,
    activation: str,
    dropout: float = 0.0,
) -> nn.Sequential:
    layers = []
    previous = input_dim
    for hidden in hidden_dims:
        layers.extend((nn.Linear(previous, hidden), _activation(activation)))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        previous = hidden
    layers.append(nn.Linear(previous, output_dim))
    return nn.Sequential(*layers)


class JointPairContextFusionNet(nn.Module):
    """Encode each current robot-entity pair directly before masked attention."""

    def __init__(self, config: Dict):
        super().__init__()
        self.config = config

        model_cfg = config["model"]
        robot_cfg = config["robot_encoding"]
        joint_cfg = config["joint_embedding"]
        reduction_cfg = config["feature_reduction"]
        attention_cfg = config["attention"]
        head_cfg = config["prediction_head"]

        self.robot_input_dim = int(model_cfg["robot_input_dim"])
        self.entity_input_dim = int(model_cfg["entity_input_dim"])
        self.return_attention = bool(model_cfg.get("return_attention", False))
        self.use_mask = bool(attention_cfg.get("use_mask", True))
        self.encode_robot_context = bool(robot_cfg.get("enabled", True))

        if self.encode_robot_context:
            self.robot_context_dim = int(robot_cfg["output_dim"])
            self.robot_projection = nn.Sequential(
                nn.Linear(self.robot_input_dim, self.robot_context_dim),
                _activation(robot_cfg["activation"]),
            )
        else:
            self.robot_context_dim = self.robot_input_dim
            self.robot_projection = nn.Identity()

        joint_dims = [int(size) for size in joint_cfg["hidden_dims"]]
        if not joint_dims:
            raise ValueError("joint_embedding.hidden_dims must contain at least one size.")
        self.interaction_dim = joint_dims[-1]
        self.joint_mlp = _build_mlp(
            input_dim=self.robot_input_dim + self.entity_input_dim,
            hidden_dims=joint_dims[:-1],
            output_dim=self.interaction_dim,
            activation=joint_cfg["activation"],
            dropout=float(joint_cfg.get("dropout", 0.0)),
        )
        self.feature_reducer = _build_mlp(
            input_dim=self.interaction_dim,
            hidden_dims=reduction_cfg["hidden_dims"],
            output_dim=reduction_cfg["output_dim"],
            activation=reduction_cfg["activation"],
            dropout=float(reduction_cfg.get("dropout", 0.0)),
        )
        self.attention_mlp = _build_mlp(
            input_dim=self.interaction_dim * 2,
            hidden_dims=attention_cfg["hidden_dims"],
            output_dim=1,
            activation=attention_cfg["activation"],
            dropout=float(attention_cfg.get("dropout", 0.0)),
        )

        self.prediction_head = None
        if bool(head_cfg.get("enabled", True)):
            self.prediction_head = _build_mlp(
                input_dim=self.robot_context_dim + int(reduction_cfg["output_dim"]),
                hidden_dims=head_cfg["hidden_dims"],
                output_dim=model_cfg["output_dim"],
                activation=head_cfg["activation"],
                dropout=float(head_cfg.get("dropout", 0.0)),
            )

        self._initialize(config.get("initialization", {}))

    @classmethod
    def from_yaml(cls, path: str) -> "JointPairContextFusionNet":
        return cls(load_architecture_config(path))

    def forward(
        self,
        robot_observation: torch.Tensor,
        entity_observations: torch.Tensor,
        entity_mask: Optional[torch.Tensor] = None,
        return_attention: Optional[bool] = None,
    ):
        current_robot = self._current_robot(robot_observation)
        current_entities = self._current_entities(entity_observations)
        robot_context = self.robot_projection(current_robot)
        batch_size, num_entities = current_entities.shape[:2]

        if entity_mask is None:
            entity_mask = torch.ones(
                batch_size, num_entities, dtype=torch.bool, device=current_entities.device
            )
        else:
            entity_mask = entity_mask.to(device=current_entities.device, dtype=torch.bool)

        if num_entities == 0:
            return self._empty_entity_forward(robot_context, return_attention)

        repeated_robot = current_robot.unsqueeze(1).expand(-1, num_entities, -1)
        interaction_embedding = self.joint_mlp(
            torch.cat((repeated_robot, current_entities), dim=-1)
        )
        global_context = self._masked_mean(interaction_embedding, entity_mask)
        attention_input = torch.cat(
            (
                interaction_embedding,
                global_context.unsqueeze(1).expand(-1, num_entities, -1),
            ),
            dim=-1,
        )
        attention_logits = self.attention_mlp(attention_input).squeeze(-1)
        attention_weights = self._masked_softmax(attention_logits, entity_mask)
        reduced_features = self.feature_reducer(interaction_embedding)
        social_context = torch.sum(
            reduced_features * attention_weights.unsqueeze(-1), dim=1
        )
        output, features = self._prediction_output(robot_context, social_context)

        should_return_attention = self.return_attention if return_attention is None else return_attention
        if not should_return_attention:
            return output if output is not None else features
        return {
            "output": output,
            "attention_weights": attention_weights,
            "robot_context": robot_context,
            "social_context": social_context,
            "interaction_embedding": interaction_embedding,
            "reduced_features": reduced_features,
        }

    @staticmethod
    def _current_robot(robot_observation: torch.Tensor) -> torch.Tensor:
        if robot_observation.dim() == 3:
            return robot_observation[:, -1, :]
        if robot_observation.dim() == 2:
            return robot_observation
        raise ValueError("Expected robot observations [batch, features] or [batch, time, features].")

    @staticmethod
    def _current_entities(entity_observations: torch.Tensor) -> torch.Tensor:
        if entity_observations.dim() == 4:
            return entity_observations[:, :, -1, :]
        if entity_observations.dim() == 3:
            return entity_observations
        raise ValueError(
            "Expected entity observations [batch, entities, features] or "
            "[batch, entities, time, features]."
        )

    def _empty_entity_forward(
        self, robot_context: torch.Tensor, return_attention: Optional[bool]
    ):
        social_dim = int(self.config["feature_reduction"]["output_dim"])
        social_context = robot_context.new_zeros((robot_context.shape[0], social_dim))
        output, features = self._prediction_output(robot_context, social_context)
        should_return_attention = self.return_attention if return_attention is None else return_attention
        if not should_return_attention:
            return output if output is not None else features
        return {
            "output": output,
            "attention_weights": robot_context.new_zeros((robot_context.shape[0], 0)),
            "robot_context": robot_context,
            "social_context": social_context,
            "interaction_embedding": robot_context.new_zeros(
                (robot_context.shape[0], 0, self.interaction_dim)
            ),
            "reduced_features": robot_context.new_zeros(
                (robot_context.shape[0], 0, social_dim)
            ),
        }

    def _prediction_output(
        self, robot_context: torch.Tensor, social_context: torch.Tensor
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        features = torch.cat((robot_context, social_context), dim=-1)
        if self.prediction_head is None:
            return None, features
        return self.prediction_head(features), features

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = mask.unsqueeze(-1).to(dtype=values.dtype)
        total = torch.sum(values * weights, dim=1)
        count = torch.clamp(weights.sum(dim=1), min=1.0)
        return total / count

    def _masked_softmax(self, logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if not self.use_mask:
            return torch.softmax(logits, dim=1)
        masked_logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        weights = torch.softmax(masked_logits, dim=1) * mask.to(dtype=logits.dtype)
        normalizer = torch.clamp(weights.sum(dim=1, keepdim=True), min=1e-8)
        return weights / normalizer

    def _initialize(self, config: Dict) -> None:
        gain = float(config.get("linear_gain", 1.0))
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=gain)
                nn.init.zeros_(module.bias)
