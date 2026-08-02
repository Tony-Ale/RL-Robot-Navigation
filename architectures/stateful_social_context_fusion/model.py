from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import torch
from torch import nn


def load_architecture_config(path: str) -> Dict:
    """Load a stateful architecture YAML file."""
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required to load architecture configs.") from exc

    with open(Path(path), "r") as file:
        return yaml.safe_load(file) or {}


def _activation(name: str) -> nn.Module:
    activations = {"relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU, "elu": nn.ELU}
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


class StatefulSocialContextFusionNet(nn.Module):
    """Stateful unidirectional form of the social-context-fusion network."""

    def __init__(self, config: Dict, robot_input_dim: int, num_entities: int):
        super().__init__()
        self.config = config
        self.num_entities = int(num_entities)
        if self.num_entities <= 0:
            raise ValueError("Stateful social context fusion requires at least one fixed entity slot.")

        model_cfg = config["model"]
        sequence_cfg = config["sequence_encoding"]
        interaction_cfg = config["interaction_embedding"]
        reduction_cfg = config["feature_reduction"]
        attention_cfg = config["attention"]

        self.robot_input_dim = int(robot_input_dim)
        self.entity_input_dim = int(model_cfg["entity_input_dim"])
        self.robot_hidden_size = int(sequence_cfg["robot_gru_hidden_size"])
        self.entity_hidden_size = int(sequence_cfg["entity_gru_hidden_size"])
        self.robot_layers = int(sequence_cfg["robot_gru_layers"])
        self.entity_layers = int(sequence_cfg["entity_gru_layers"])
        self.use_mask = bool(attention_cfg.get("use_mask", True))
        self.last_attention_weights: Optional[torch.Tensor] = None

        dropout = float(sequence_cfg.get("dropout", 0.0))
        self.robot_gru = nn.GRU(
            self.robot_input_dim,
            self.robot_hidden_size,
            num_layers=self.robot_layers,
            dropout=dropout if self.robot_layers > 1 else 0.0,
        )
        self.entity_gru = nn.GRU(
            self.entity_input_dim,
            self.entity_hidden_size,
            num_layers=self.entity_layers,
            dropout=dropout if self.entity_layers > 1 else 0.0,
        )

        interaction_dim = int(interaction_cfg["hidden_dims"][-1])
        self.interaction_mlp = _build_mlp(
            self.robot_hidden_size + self.entity_hidden_size,
            interaction_cfg["hidden_dims"][:-1],
            interaction_dim,
            interaction_cfg["activation"],
            float(interaction_cfg.get("dropout", 0.0)),
        )
        self.feature_reducer = _build_mlp(
            interaction_dim,
            reduction_cfg["hidden_dims"],
            int(reduction_cfg["output_dim"]),
            reduction_cfg["activation"],
            float(reduction_cfg.get("dropout", 0.0)),
        )
        self.attention_mlp = _build_mlp(
            interaction_dim * 2,
            attention_cfg["hidden_dims"],
            1,
            attention_cfg["activation"],
            float(attention_cfg.get("dropout", 0.0)),
        )

        self.output_dim = self.robot_hidden_size + int(reduction_cfg["output_dim"])
        self.state_size = (
            self.robot_layers * self.robot_hidden_size
            + self.entity_layers * self.num_entities * self.entity_hidden_size
        )
        self._initialize(config.get("initialization", {}))

    def forward_sequence(
        self,
        robot: torch.Tensor,
        entities: torch.Tensor,
        entity_mask: torch.Tensor,
        packed_state: torch.Tensor,
        episode_starts: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process ordered sequences and return fused features plus final recurrent state."""
        if packed_state.dim() != 3 or packed_state.shape[0] != 1 or packed_state.shape[-1] != self.state_size:
            raise ValueError(f"Expected packed state [1, sequences, {self.state_size}].")

        n_sequences = packed_state.shape[1]
        robot_sequence = self._as_sequence(robot, n_sequences)
        entity_sequence = self._as_sequence(entities, n_sequences)
        mask_sequence = self._as_sequence(entity_mask, n_sequences)
        start_sequence = self._as_sequence(episode_starts, n_sequences)
        robot_state, entity_state = self._unpack_state(packed_state)

        outputs = []
        for robot_step, entity_step, mask_step, starts in zip(
            robot_sequence, entity_sequence, mask_sequence, start_sequence
        ):
            keep = (1.0 - starts.float()).view(1, n_sequences, 1)
            robot_state = robot_state * keep
            entity_state = entity_state * keep.unsqueeze(2)

            robot_output, robot_state = self.robot_gru(robot_step.unsqueeze(0), robot_state)
            flat_entities = entity_step.reshape(n_sequences * self.num_entities, self.entity_input_dim)
            flat_entity_state = entity_state.reshape(
                self.entity_layers, n_sequences * self.num_entities, self.entity_hidden_size
            )
            valid_state = mask_step.reshape(1, n_sequences * self.num_entities, 1).to(flat_entity_state.dtype)
            flat_entity_state = flat_entity_state * valid_state
            entity_output, flat_entity_state = self.entity_gru(flat_entities.unsqueeze(0), flat_entity_state)
            flat_entity_state = flat_entity_state * valid_state
            entity_state = flat_entity_state.reshape(
                self.entity_layers, n_sequences, self.num_entities, self.entity_hidden_size
            )
            entity_context = entity_output.squeeze(0).reshape(
                n_sequences, self.num_entities, self.entity_hidden_size
            )
            outputs.append(self._fuse(robot_output.squeeze(0), entity_context, mask_step))

        features = torch.stack(outputs).transpose(0, 1).reshape(-1, self.output_dim)
        return features, self._pack_state(robot_state, entity_state)

    @staticmethod
    def _as_sequence(values: torch.Tensor, n_sequences: int) -> torch.Tensor:
        return values.reshape(n_sequences, -1, *values.shape[1:]).transpose(0, 1)

    def _fuse(
        self,
        robot_context: torch.Tensor,
        entity_context: torch.Tensor,
        entity_mask: torch.Tensor,
    ) -> torch.Tensor:
        repeated_robot = robot_context.unsqueeze(1).expand(-1, self.num_entities, -1)
        interaction = self.interaction_mlp(torch.cat((repeated_robot, entity_context), dim=-1))
        global_context = self._masked_mean(interaction, entity_mask)
        attention_input = torch.cat(
            (interaction, global_context.unsqueeze(1).expand(-1, self.num_entities, -1)), dim=-1
        )
        logits = self.attention_mlp(attention_input).squeeze(-1)
        weights = self._masked_softmax(logits, entity_mask)
        self.last_attention_weights = weights.detach()
        social_context = torch.sum(self.feature_reducer(interaction) * weights.unsqueeze(-1), dim=1)
        return torch.cat((robot_context, social_context), dim=-1)

    def _unpack_state(self, packed: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = packed.shape[1]
        flat = packed.squeeze(0)
        robot_count = self.robot_layers * self.robot_hidden_size
        robot = flat[:, :robot_count].reshape(batch_size, self.robot_layers, self.robot_hidden_size)
        entities = flat[:, robot_count:].reshape(
            batch_size, self.entity_layers, self.num_entities, self.entity_hidden_size
        )
        return robot.transpose(0, 1).contiguous(), entities.permute(1, 0, 2, 3).contiguous()

    @staticmethod
    def _pack_state(robot: torch.Tensor, entities: torch.Tensor) -> torch.Tensor:
        batch_size = robot.shape[1]
        robot_flat = robot.transpose(0, 1).reshape(batch_size, -1)
        entity_flat = entities.permute(1, 0, 2, 3).reshape(batch_size, -1)
        return torch.cat((robot_flat, entity_flat), dim=-1).unsqueeze(0)

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = mask.unsqueeze(-1).to(values.dtype)
        return torch.sum(values * weights, dim=1) / torch.clamp(weights.sum(dim=1), min=1.0)

    def _masked_softmax(self, logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if not self.use_mask:
            return torch.softmax(logits, dim=1)
        masked = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        weights = torch.softmax(masked, dim=1) * mask.to(logits.dtype)
        return weights / torch.clamp(weights.sum(dim=1, keepdim=True), min=1e-8)

    def _initialize(self, config: Dict) -> None:
        gain = float(config.get("linear_gain", 1.0))
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=gain)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.GRU) and config.get("orthogonal_gru", True):
                for name, parameter in module.named_parameters():
                    if "weight_hh" in name:
                        nn.init.orthogonal_(parameter)
                    elif "weight_ih" in name:
                        nn.init.xavier_uniform_(parameter)
                    elif "bias" in name:
                        nn.init.zeros_(parameter)
