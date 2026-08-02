from typing import Dict, Optional

import torch
from torch import nn

from architectures.social_context_fusion.model import (
    SocialContextFusionNet,
    _build_mlp,
    load_architecture_config,
)


class HybridContextFusionNet(SocialContextFusionNet):
    """Social-context fusion with recurrent humans and current-state static entities."""

    HUMAN_TYPE_INDEX = 1

    def __init__(self, config: Dict):
        super().__init__(config)

        static_cfg = config["static_entity_embedding"]
        entity_dim = int(config["model"]["entity_input_dim"])
        static_output_dim = int(static_cfg["output_dim"])
        if entity_dim <= self.HUMAN_TYPE_INDEX:
            raise ValueError("Entity observations do not contain the SocNavGym human type feature.")
        if static_output_dim != self.entity_encoder.output_dim:
            raise ValueError(
                "static_entity_embedding.output_dim must equal the human BiGRU output dimension "
                f"({self.entity_encoder.output_dim})."
            )

        # Normalize the recurrent and feed-forward branches independently before
        # their contexts enter the shared interaction and attention networks.
        self.human_context_norm = nn.LayerNorm(self.entity_encoder.output_dim)
        self.static_context_norm = nn.LayerNorm(static_output_dim)
        self.static_entity_encoder = _build_mlp(
            input_dim=entity_dim,
            hidden_dims=static_cfg["hidden_dims"],
            output_dim=static_output_dim,
            activation=static_cfg["activation"],
            dropout=static_cfg["dropout"],
        )
        self._initialize_static_encoder(config.get("initialization", {}))

    @classmethod
    def from_yaml(cls, path: str) -> "HybridContextFusionNet":
        return cls(load_architecture_config(path))

    def _encode_entities(
        self,
        entity_history: torch.Tensor,
        entity_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if entity_history.dim() == 3:
            entity_history = entity_history.unsqueeze(2)
        if entity_history.dim() != 4:
            raise ValueError(
                "Expected entity_history shape [batch, entities, time, features] "
                "or [batch, entities, features]."
            )

        batch_size, num_entities, time_steps, feature_dim = entity_history.shape
        output_dim = self.entity_encoder.output_dim
        if num_entities == 0:
            return entity_history.new_zeros((batch_size, 0, output_dim))

        flat_history = entity_history.reshape(batch_size * num_entities, time_steps, feature_dim)
        current_entities = flat_history[:, -1, :]
        if entity_mask is None:
            valid = torch.ones(batch_size * num_entities, dtype=torch.bool, device=entity_history.device)
        else:
            valid = entity_mask.reshape(-1).to(device=entity_history.device, dtype=torch.bool)

        is_human = current_entities[:, self.HUMAN_TYPE_INDEX] > 0.5
        human_indices = torch.nonzero(valid & is_human, as_tuple=False).squeeze(-1)
        static_indices = torch.nonzero(valid & ~is_human, as_tuple=False).squeeze(-1)

        encoded = entity_history.new_zeros((batch_size * num_entities, output_dim))
        if human_indices.numel() > 0:
            human_context = self.entity_encoder(flat_history.index_select(0, human_indices))
            human_context = self.human_context_norm(human_context)
            encoded = encoded.index_copy(0, human_indices, human_context)
        if static_indices.numel() > 0:
            static_context = self.static_entity_encoder(current_entities.index_select(0, static_indices))
            static_context = self.static_context_norm(static_context)
            encoded = encoded.index_copy(0, static_indices, static_context)

        return encoded.reshape(batch_size, num_entities, output_dim)

    def _initialize_static_encoder(self, init_cfg: Dict) -> None:
        gain = float(init_cfg.get("linear_gain", 1.0))
        for module in self.static_entity_encoder.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=gain)
                nn.init.zeros_(module.bias)
