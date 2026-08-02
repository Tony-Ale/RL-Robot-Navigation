from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import torch
from torch import nn


def load_architecture_config(path: str) -> Dict:
    """Load a YAML architecture config."""
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required to load architecture configs.") from exc

    with open(Path(path), "r") as f:
        return yaml.safe_load(f) or {}


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
    final_activation: Optional[nn.Module] = None,
) -> nn.Sequential:
    layers = []
    previous = input_dim
    for hidden in hidden_dims:
        layers.append(nn.Linear(previous, hidden))
        layers.append(_activation(activation))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        previous = hidden

    layers.append(nn.Linear(previous, output_dim))
    if final_activation is not None:
        layers.append(final_activation)
    return nn.Sequential(*layers)


class SequenceBiGRUEncoder(nn.Module):
    """Encode an entity sequence and return the concatenated final forward/backward states."""

    def __init__(self, input_dim: int, hidden_size: int, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.output_dim = hidden_size * 2
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        if sequence.dim() == 2:
            sequence = sequence.unsqueeze(1)
        if sequence.dim() != 3:
            raise ValueError("Expected sequence shape [batch, time, features] or [batch, features].")

        _, hidden = self.gru(sequence)
        forward = hidden[-2]
        backward = hidden[-1]
        return torch.cat([forward, backward], dim=-1)


class CrowdContextFusionNet(nn.Module):
    """
    Crowd-BiGRU plus robot-MLP social-attention architecture.

    Input shapes:
        robot_observation: [batch, robot_input_dim] or [batch, robot_time, robot_input_dim]
        entity_history: [batch, entities, entity_time, entity_input_dim]
                        or [batch, entities, entity_input_dim]
        entity_mask: [batch, entities], True for valid entities and False for padding.

    The robot observation is embedded with an MLP, while each human/entity sequence is
    encoded independently by a shared BiGRU crowd encoder.
    """

    def __init__(self, config: Dict):
        super().__init__()
        self.config = config

        model_cfg = config["model"]
        crowd_cfg = config["crowd_encoding"]
        robot_cfg = config["robot_embedding"]
        interaction_cfg = config["interaction_embedding"]
        reduction_cfg = config["feature_reduction"]
        attention_cfg = config["attention"]
        head_cfg = config["prediction_head"]

        self.robot_input_dim = model_cfg["robot_input_dim"]
        self.entity_input_dim = model_cfg["entity_input_dim"]
        self.return_attention = bool(model_cfg.get("return_attention", False))
        self.use_mask = bool(attention_cfg.get("use_mask", True))

        self.robot_embedder = _build_mlp(
            input_dim=self.robot_input_dim,
            hidden_dims=robot_cfg["hidden_dims"],
            output_dim=robot_cfg["output_dim"],
            activation=robot_cfg["activation"],
            dropout=robot_cfg["dropout"],
        )
        self.crowd_encoder = SequenceBiGRUEncoder(
            input_dim=self.entity_input_dim,
            hidden_size=crowd_cfg["bigru_hidden_size"],
            num_layers=crowd_cfg["bigru_layers"],
            dropout=crowd_cfg["dropout"],
        )

        interaction_input_dim = robot_cfg["output_dim"] + self.crowd_encoder.output_dim
        interaction_output_dim = interaction_cfg["hidden_dims"][-1]
        self.interaction_mlp = _build_mlp(
            input_dim=interaction_input_dim,
            hidden_dims=interaction_cfg["hidden_dims"][:-1],
            output_dim=interaction_output_dim,
            activation=interaction_cfg["activation"],
            dropout=interaction_cfg["dropout"],
        )

        self.feature_reducer = _build_mlp(
            input_dim=interaction_output_dim,
            hidden_dims=reduction_cfg["hidden_dims"],
            output_dim=reduction_cfg["output_dim"],
            activation=reduction_cfg["activation"],
            dropout=reduction_cfg["dropout"],
        )

        self.attention_mlp = _build_mlp(
            input_dim=interaction_output_dim * 2,
            hidden_dims=attention_cfg["hidden_dims"],
            output_dim=1,
            activation=attention_cfg["activation"],
            dropout=attention_cfg["dropout"],
        )

        head_input_dim = robot_cfg["output_dim"] + reduction_cfg["output_dim"]
        self.prediction_head = None
        if bool(head_cfg.get("enabled", True)):
            self.prediction_head = _build_mlp(
                input_dim=head_input_dim,
                hidden_dims=head_cfg["hidden_dims"],
                output_dim=model_cfg["output_dim"],
                activation=head_cfg["activation"],
                dropout=head_cfg["dropout"],
            )

        self._initialize(config.get("initialization", {}))

    @classmethod
    def from_yaml(cls, path: str) -> "CrowdContextFusionNet":
        return cls(load_architecture_config(path))

    def forward(
        self,
        robot_observation: torch.Tensor,
        entity_history: torch.Tensor,
        entity_mask: Optional[torch.Tensor] = None,
        return_attention: Optional[bool] = None,
    ):
        robot_single = self._robot_single_observation(robot_observation)
        robot_context = self.robot_embedder(robot_single)
        crowd_context = self._encode_entities(entity_history)

        batch_size, num_entities, _ = crowd_context.shape
        if entity_mask is None:
            entity_mask = torch.ones(batch_size, num_entities, dtype=torch.bool, device=crowd_context.device)
        else:
            entity_mask = entity_mask.to(device=crowd_context.device, dtype=torch.bool)

        if num_entities == 0:
            return self._empty_entity_forward(robot_context, return_attention)

        repeated_robot = robot_context.unsqueeze(1).expand(-1, num_entities, -1)
        joint_features = torch.cat([repeated_robot, crowd_context], dim=-1)
        interaction_embedding = self.interaction_mlp(joint_features)

        global_context = self._masked_mean(interaction_embedding, entity_mask)
        attention_input = torch.cat(
            [interaction_embedding, global_context.unsqueeze(1).expand(-1, num_entities, -1)],
            dim=-1,
        )
        attention_logits = self.attention_mlp(attention_input).squeeze(-1)
        attention_weights = self._masked_softmax(attention_logits, entity_mask)

        reduced_features = self.feature_reducer(interaction_embedding)
        social_context = torch.sum(reduced_features * attention_weights.unsqueeze(-1), dim=1)

        output, features = self._prediction_output(robot_context, social_context)

        should_return_attention = self.return_attention if return_attention is None else return_attention
        if not should_return_attention:
            return output if output is not None else features

        return {
            "output": output,
            "attention_weights": attention_weights,
            "robot_context": robot_context,
            "crowd_context": crowd_context,
            "social_context": social_context,
            "interaction_embedding": interaction_embedding,
            "reduced_features": reduced_features,
        }

    def _robot_single_observation(self, robot_observation: torch.Tensor) -> torch.Tensor:
        if robot_observation.dim() == 2:
            return robot_observation
        if robot_observation.dim() == 3:
            return robot_observation[:, -1, :]
        raise ValueError("Expected robot_observation shape [batch, features] or [batch, time, features].")

    def _encode_entities(self, entity_history: torch.Tensor) -> torch.Tensor:
        if entity_history.dim() == 3:
            entity_history = entity_history.unsqueeze(2)
        if entity_history.dim() != 4:
            raise ValueError("Expected entity_history shape [batch, entities, time, features] or [batch, entities, features].")

        batch_size, num_entities, time_steps, feature_dim = entity_history.shape
        if num_entities == 0:
            return entity_history.new_zeros((batch_size, 0, self.crowd_encoder.output_dim))
        if feature_dim != self.entity_input_dim:
            raise ValueError(f"Expected entity feature dim {self.entity_input_dim}, got {feature_dim}.")

        flat = entity_history.reshape(batch_size * num_entities, time_steps, feature_dim)
        encoded = self.crowd_encoder(flat)
        return encoded.reshape(batch_size, num_entities, self.crowd_encoder.output_dim)

    def _empty_entity_forward(self, robot_context: torch.Tensor, return_attention: Optional[bool]):
        social_dim = self.config["feature_reduction"]["output_dim"]
        interaction_dim = self.config["interaction_embedding"]["hidden_dims"][-1]
        social_context = robot_context.new_zeros((robot_context.shape[0], social_dim))
        output, features = self._prediction_output(robot_context, social_context)
        should_return_attention = self.return_attention if return_attention is None else return_attention
        if not should_return_attention:
            return output if output is not None else features
        return {
            "output": output,
            "attention_weights": robot_context.new_zeros((robot_context.shape[0], 0)),
            "robot_context": robot_context,
            "crowd_context": robot_context.new_zeros((robot_context.shape[0], 0, self.crowd_encoder.output_dim)),
            "social_context": social_context,
            "interaction_embedding": robot_context.new_zeros((robot_context.shape[0], 0, interaction_dim)),
            "reduced_features": robot_context.new_zeros((robot_context.shape[0], 0, social_dim)),
        }

    def _prediction_output(self, robot_context: torch.Tensor, social_context: torch.Tensor) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        features = torch.cat([robot_context, social_context], dim=-1)
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
        weights = torch.softmax(masked_logits, dim=1)
        weights = weights * mask.to(dtype=weights.dtype)
        normalizer = torch.clamp(weights.sum(dim=1, keepdim=True), min=1e-8)
        return weights / normalizer

    def _initialize(self, init_cfg: Dict) -> None:
        linear_gain = float(init_cfg.get("linear_gain", 1.0))
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=linear_gain)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.GRU) and init_cfg.get("orthogonal_gru", True):
                for name, parameter in module.named_parameters():
                    if "weight_hh" in name:
                        nn.init.orthogonal_(parameter)
                    elif "weight_ih" in name:
                        nn.init.xavier_uniform_(parameter)
                    elif "bias" in name:
                        nn.init.zeros_(parameter)
