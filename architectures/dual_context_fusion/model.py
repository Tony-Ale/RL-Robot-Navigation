from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import torch
from torch import nn


def load_architecture_config(path: str) -> Dict:
    """Load a dual-context architecture YAML file."""
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required to load architecture configs.") from exc

    with open(Path(path), "r") as config_file:
        return yaml.safe_load(config_file) or {}


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
    previous_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(previous_dim, hidden_dim))
        layers.append(_activation(activation))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        previous_dim = hidden_dim
    layers.append(nn.Linear(previous_dim, output_dim))
    return nn.Sequential(*layers)


class SequenceBiGRUEncoder(nn.Module):
    """Encode a sequence using its final forward and backward GRU states."""

    def __init__(self, input_dim: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
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
        return torch.cat([hidden[-2], hidden[-1]], dim=-1)


class DualContextFusionNet(nn.Module):
    """Fuse recurrent human context and current static-obstacle context independently."""

    HUMAN_TYPE_INDEX = 1

    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        model_cfg = config["model"]
        sequence_cfg = config["sequence_encoding"]
        static_cfg = config["static_entity_embedding"]
        human_interaction_cfg = config["human_interaction_embedding"]
        static_interaction_cfg = config["static_interaction_embedding"]
        human_reduction_cfg = config["human_feature_reduction"]
        static_reduction_cfg = config["static_feature_reduction"]
        human_attention_cfg = config["human_attention"]
        static_attention_cfg = config["static_attention"]
        head_cfg = config["prediction_head"]

        self.return_attention = bool(model_cfg.get("return_attention", False))
        entity_input_dim = int(model_cfg["entity_input_dim"])
        if entity_input_dim <= self.HUMAN_TYPE_INDEX:
            raise ValueError("Entity observations do not contain the SocNavGym human type feature.")

        self.robot_encoder = SequenceBiGRUEncoder(
            input_dim=int(model_cfg["robot_input_dim"]),
            hidden_size=int(sequence_cfg["robot_gru_hidden_size"]),
            num_layers=int(sequence_cfg["robot_gru_layers"]),
            dropout=float(sequence_cfg["robot_dropout"]),
        )
        self.human_encoder = SequenceBiGRUEncoder(
            input_dim=entity_input_dim,
            hidden_size=int(sequence_cfg["human_gru_hidden_size"]),
            num_layers=int(sequence_cfg["human_gru_layers"]),
            dropout=float(sequence_cfg["human_dropout"]),
        )
        static_output_dim = int(static_cfg["output_dim"])
        self.static_entity_encoder = _build_mlp(
            input_dim=entity_input_dim,
            hidden_dims=static_cfg["hidden_dims"],
            output_dim=static_output_dim,
            activation=static_cfg["activation"],
            dropout=float(static_cfg["dropout"]),
        )
        self.human_context_norm = nn.LayerNorm(self.human_encoder.output_dim)
        self.static_context_norm = nn.LayerNorm(static_output_dim)

        human_interaction_dim = self._interaction_output_dim(human_interaction_cfg, "human_interaction_embedding")
        static_interaction_dim = self._interaction_output_dim(static_interaction_cfg, "static_interaction_embedding")
        self.human_interaction_mlp = self._build_interaction_mlp(
            self.robot_encoder.output_dim + self.human_encoder.output_dim,
            human_interaction_cfg,
        )
        self.static_interaction_mlp = self._build_interaction_mlp(
            self.robot_encoder.output_dim + static_output_dim,
            static_interaction_cfg,
        )
        self.human_feature_reducer = _build_mlp(
            input_dim=human_interaction_dim,
            hidden_dims=human_reduction_cfg["hidden_dims"],
            output_dim=int(human_reduction_cfg["output_dim"]),
            activation=human_reduction_cfg["activation"],
            dropout=float(human_reduction_cfg["dropout"]),
        )
        self.static_feature_reducer = _build_mlp(
            input_dim=static_interaction_dim,
            hidden_dims=static_reduction_cfg["hidden_dims"],
            output_dim=int(static_reduction_cfg["output_dim"]),
            activation=static_reduction_cfg["activation"],
            dropout=float(static_reduction_cfg["dropout"]),
        )
        self.human_attention_mlp = _build_mlp(
            input_dim=human_interaction_dim * 2,
            hidden_dims=human_attention_cfg["hidden_dims"],
            output_dim=1,
            activation=human_attention_cfg["activation"],
            dropout=float(human_attention_cfg["dropout"]),
        )
        self.static_attention_mlp = _build_mlp(
            input_dim=static_interaction_dim * 2,
            hidden_dims=static_attention_cfg["hidden_dims"],
            output_dim=1,
            activation=static_attention_cfg["activation"],
            dropout=float(static_attention_cfg["dropout"]),
        )

        head_input_dim = (
            self.robot_encoder.output_dim
            + int(human_reduction_cfg["output_dim"])
            + int(static_reduction_cfg["output_dim"])
        )
        self.prediction_head = None
        if bool(head_cfg.get("enabled", True)):
            self.prediction_head = _build_mlp(
                input_dim=head_input_dim,
                hidden_dims=head_cfg["hidden_dims"],
                output_dim=int(model_cfg["output_dim"]),
                activation=head_cfg["activation"],
                dropout=float(head_cfg["dropout"]),
            )

        self._initialize(config.get("initialization", {}))

    @classmethod
    def from_yaml(cls, path: str) -> "DualContextFusionNet":
        return cls(load_architecture_config(path))

    def forward(
        self,
        robot_history: torch.Tensor,
        entity_history: torch.Tensor,
        entity_mask: Optional[torch.Tensor] = None,
        return_attention: Optional[bool] = None,
    ):
        if entity_history.dim() == 3:
            entity_history = entity_history.unsqueeze(2)
        if entity_history.dim() != 4:
            raise ValueError(
                "Expected entity_history shape [batch, entities, time, features] "
                "or [batch, entities, features]."
            )

        robot_context = self.robot_encoder(robot_history)
        batch_size, num_entities = entity_history.shape[:2]
        valid_mask = self._validated_mask(entity_mask, batch_size, num_entities, entity_history.device)
        current_entities = entity_history[:, :, -1, :]
        human_mask = valid_mask & (current_entities[:, :, self.HUMAN_TYPE_INDEX] > 0.5)
        static_mask = valid_mask & ~human_mask

        human_representations = self._encode_humans(entity_history, human_mask)
        static_representations = self._encode_static(current_entities, static_mask)
        human_context, human_weights, human_interactions, human_reduced = self._pool_branch(
            robot_context,
            human_representations,
            human_mask,
            self.human_interaction_mlp,
            self.human_feature_reducer,
            self.human_attention_mlp,
            int(self.config["human_feature_reduction"]["output_dim"]),
        )
        obstacle_context, static_weights, static_interactions, static_reduced = self._pool_branch(
            robot_context,
            static_representations,
            static_mask,
            self.static_interaction_mlp,
            self.static_feature_reducer,
            self.static_attention_mlp,
            int(self.config["static_feature_reduction"]["output_dim"]),
        )

        social_context = torch.cat([human_context, obstacle_context], dim=-1)
        features = torch.cat([robot_context, social_context], dim=-1)
        output = self.prediction_head(features) if self.prediction_head is not None else None
        should_return_attention = self.return_attention if return_attention is None else return_attention
        if not should_return_attention:
            return output if output is not None else features

        return {
            "output": output,
            "robot_context": robot_context,
            "human_context": human_context,
            "obstacle_context": obstacle_context,
            "social_context": social_context,
            "human_attention_weights": human_weights,
            "static_attention_weights": static_weights,
            "human_interaction_embedding": human_interactions,
            "static_interaction_embedding": static_interactions,
            "human_reduced_features": human_reduced,
            "static_reduced_features": static_reduced,
        }

    def _encode_humans(self, entity_history: torch.Tensor, human_mask: torch.Tensor) -> torch.Tensor:
        batch_size, num_entities, time_steps, feature_dim = entity_history.shape
        encoded = entity_history.new_zeros((batch_size * num_entities, self.human_encoder.output_dim))
        indices = torch.nonzero(human_mask.reshape(-1), as_tuple=False).squeeze(-1)
        if indices.numel() > 0:
            histories = entity_history.reshape(batch_size * num_entities, time_steps, feature_dim)
            contexts = self.human_context_norm(self.human_encoder(histories.index_select(0, indices)))
            encoded = encoded.index_copy(0, indices, contexts)
        return encoded.reshape(batch_size, num_entities, self.human_encoder.output_dim)

    def _encode_static(self, current_entities: torch.Tensor, static_mask: torch.Tensor) -> torch.Tensor:
        batch_size, num_entities, feature_dim = current_entities.shape
        output_dim = int(self.config["static_entity_embedding"]["output_dim"])
        encoded = current_entities.new_zeros((batch_size * num_entities, output_dim))
        indices = torch.nonzero(static_mask.reshape(-1), as_tuple=False).squeeze(-1)
        if indices.numel() > 0:
            current = current_entities.reshape(batch_size * num_entities, feature_dim)
            contexts = self.static_context_norm(self.static_entity_encoder(current.index_select(0, indices)))
            encoded = encoded.index_copy(0, indices, contexts)
        return encoded.reshape(batch_size, num_entities, output_dim)

    @staticmethod
    def _pool_branch(
        robot_context: torch.Tensor,
        entity_context: torch.Tensor,
        branch_mask: torch.Tensor,
        interaction_mlp: nn.Module,
        feature_reducer: nn.Module,
        attention_mlp: nn.Module,
        output_dim: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, num_entities = entity_context.shape[:2]
        if num_entities == 0:
            empty_weights = robot_context.new_zeros((batch_size, 0))
            interaction_dim = interaction_mlp[-1].out_features
            empty_interactions = robot_context.new_zeros((batch_size, 0, interaction_dim))
            empty_reduced = robot_context.new_zeros((batch_size, 0, output_dim))
            return robot_context.new_zeros((batch_size, output_dim)), empty_weights, empty_interactions, empty_reduced

        repeated_robot = robot_context.unsqueeze(1).expand(-1, num_entities, -1)
        interactions = interaction_mlp(torch.cat([repeated_robot, entity_context], dim=-1))
        global_context = DualContextFusionNet._masked_mean(interactions, branch_mask)
        attention_input = torch.cat(
            [interactions, global_context.unsqueeze(1).expand(-1, num_entities, -1)],
            dim=-1,
        )
        attention_weights = DualContextFusionNet._masked_softmax(
            attention_mlp(attention_input).squeeze(-1),
            branch_mask,
        )
        reduced_features = feature_reducer(interactions)
        pooled_context = torch.sum(reduced_features * attention_weights.unsqueeze(-1), dim=1)
        return pooled_context, attention_weights, interactions, reduced_features

    @staticmethod
    def _validated_mask(entity_mask, batch_size, num_entities, device):
        if entity_mask is None:
            return torch.ones(batch_size, num_entities, dtype=torch.bool, device=device)
        mask = entity_mask.to(device=device, dtype=torch.bool)
        if tuple(mask.shape) != (batch_size, num_entities):
            raise ValueError(f"Expected entity_mask shape {(batch_size, num_entities)}, got {tuple(mask.shape)}.")
        return mask

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = mask.unsqueeze(-1).to(dtype=values.dtype)
        return torch.sum(values * weights, dim=1) / torch.clamp(weights.sum(dim=1), min=1.0)

    @staticmethod
    def _masked_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        masked_logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        weights = torch.softmax(masked_logits, dim=1) * mask.to(dtype=logits.dtype)
        return weights / torch.clamp(weights.sum(dim=1, keepdim=True), min=1e-8)

    @staticmethod
    def _interaction_output_dim(config: Dict, name: str) -> int:
        hidden_dims = config.get("hidden_dims", [])
        if not hidden_dims:
            raise ValueError(f"{name}.hidden_dims must contain at least one dimension.")
        return int(hidden_dims[-1])

    @staticmethod
    def _build_interaction_mlp(input_dim: int, config: Dict) -> nn.Sequential:
        hidden_dims = config["hidden_dims"]
        return _build_mlp(
            input_dim=input_dim,
            hidden_dims=hidden_dims[:-1],
            output_dim=int(hidden_dims[-1]),
            activation=config["activation"],
            dropout=float(config["dropout"]),
        )

    def _initialize(self, config: Dict) -> None:
        linear_gain = float(config.get("linear_gain", 1.0))
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=linear_gain)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.GRU) and config.get("orthogonal_gru", True):
                for name, parameter in module.named_parameters():
                    if "weight_hh" in name:
                        nn.init.orthogonal_(parameter)
                    elif "weight_ih" in name:
                        nn.init.xavier_uniform_(parameter)
                    elif "bias" in name:
                        nn.init.zeros_(parameter)
