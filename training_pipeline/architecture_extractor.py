from copy import deepcopy
from typing import Dict, Iterable, List, Optional

from training_pipeline.utils import configure_matplotlib_cache, load_yaml

configure_matplotlib_cache()

import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


ARCHITECTURE_REGISTRY = {
    "social_context_fusion": ("architectures.social_context_fusion", "SocialContextFusionNet"),
    "feedforward_social_context_fusion": (
        "architectures.feedforward_social_context_fusion",
        "FeedForwardSocialContextFusionNet",
    ),
    "joint_pair_context_fusion": (
        "architectures.joint_pair_context_fusion",
        "JointPairContextFusionNet",
    ),
    "hybrid_context_fusion": ("architectures.hybrid_context_fusion", "HybridContextFusionNet"),
    "dual_context_fusion": ("architectures.dual_context_fusion", "DualContextFusionNet"),
    "joint_scene_fusion": ("architectures.joint_scene_fusion", "JointSceneFusionNet"),
    "crowd_context_fusion": ("architectures.crowd_context_fusion", "CrowdContextFusionNet"),
}


def load_architecture(name: str, config_path: str, robot_input_dim: Optional[int] = None):
    """Instantiate one of the implemented architecture variants."""
    if name not in ARCHITECTURE_REGISTRY:
        valid = ", ".join(sorted(ARCHITECTURE_REGISTRY))
        raise ValueError(f"Unknown architecture '{name}'. Expected one of: {valid}.")

    import importlib

    module_name, class_name = ARCHITECTURE_REGISTRY[name]
    module = importlib.import_module(module_name)
    architecture_cls = getattr(module, class_name)
    config = load_yaml(config_path)
    if robot_input_dim is not None:
        config = deepcopy(config)
        config["model"]["robot_input_dim"] = int(robot_input_dim)
    return architecture_cls(config)


def architecture_feature_dim(
    name: str,
    architecture_config: Dict,
    effective_robot_dim: Optional[int] = None,
) -> int:
    """Return the PPO feature dimension produced from architecture internals."""
    if name == "dual_context_fusion":
        robot_dim = int(architecture_config["sequence_encoding"]["robot_gru_hidden_size"]) * 2
        human_dim = int(architecture_config["human_feature_reduction"]["output_dim"])
        static_dim = int(architecture_config["static_feature_reduction"]["output_dim"])
        return robot_dim + human_dim + static_dim

    reduction_dim = int(architecture_config["feature_reduction"]["output_dim"])
    if name == "feedforward_social_context_fusion":
        robot_dim = int(architecture_config["observation_encoding"]["robot_output_dim"])
    elif name == "joint_pair_context_fusion":
        robot_cfg = architecture_config["robot_encoding"]
        robot_dim = (
            int(robot_cfg["output_dim"])
            if robot_cfg.get("enabled", True)
            else int(effective_robot_dim or architecture_config["model"]["robot_input_dim"])
        )
    elif name in ("social_context_fusion", "hybrid_context_fusion"):
        robot_dim = int(architecture_config["sequence_encoding"]["robot_gru_hidden_size"]) * 2
    elif name == "joint_scene_fusion":
        robot_dim = int(architecture_config["robot_projection"]["output_dim"])
    elif name == "crowd_context_fusion":
        robot_dim = int(architecture_config["robot_embedding"]["output_dim"])
    else:
        raise ValueError(f"Unknown architecture '{name}'.")
    return robot_dim + reduction_dim


def _concat_entity_tensors(
    observations: Dict[str, torch.Tensor],
    entity_keys: Iterable[str],
    entity_feature_dim: int,
) -> torch.Tensor:
    entity_batches: List[torch.Tensor] = []
    batch_size = observations["robot"].shape[0]
    device = observations["robot"].device

    for key in entity_keys:
        if key not in observations:
            continue
        values = observations[key].float()
        if values.dim() == 4 and values.shape[-1] == entity_feature_dim:
            entity_batches.append(values)
        elif values.dim() == 3 and values.shape[-1] == entity_feature_dim:
            entity_batches.append(values)
        elif values.dim() == 2 and values.shape[-1] % entity_feature_dim == 0:
            entity_batches.append(values.reshape(values.shape[0], -1, entity_feature_dim))
        else:
            raise ValueError(
                f"Observation key '{key}' must contain {entity_feature_dim}-value entity rows, "
                f"got {tuple(values.shape)}."
            )

    if entity_batches:
        return torch.cat(entity_batches, dim=1)
    return torch.zeros((batch_size, 0, entity_feature_dim), dtype=torch.float32, device=device)


def _entity_mask(entities: torch.Tensor, mask_zero_entities: bool) -> torch.Tensor:
    if not mask_zero_entities:
        return torch.ones(entities.shape[:2], dtype=torch.bool, device=entities.device)
    current_entities = entities[:, :, -1, :] if entities.dim() == 4 else entities
    return torch.sum(torch.abs(current_entities), dim=-1) > 1e-8


def _space_dim(observation_space, key: str) -> int:
    if not hasattr(observation_space, "spaces") or key not in observation_space.spaces:
        return 0
    shape = observation_space.spaces[key].shape
    robot_shape = observation_space.spaces["robot"].shape
    history_length = robot_shape[0] if len(robot_shape) > 1 else None
    if history_length is not None and shape and shape[0] == history_length:
        shape = shape[1:]
    return int(torch.tensor(shape).prod().item()) if shape else 0


def effective_robot_input_dim(observation_space, include_waypoint_features: bool = False) -> int:
    """Return the robot input size the architecture will actually receive."""
    robot_dim = _space_dim(observation_space, "robot")
    waypoint_dim = 0
    if include_waypoint_features:
        waypoint_dim = _space_dim(observation_space, "waypoint_features")
    return robot_dim + waypoint_dim


class ArchitectureFeaturesExtractor(BaseFeaturesExtractor):
    """
    Stable-Baselines3 feature extractor backed by the implemented architectures.

    The architecture produces robot and social context tensors. PPO receives
    their concatenation and then learns its own actor/critic heads.
    """

    def __init__(
        self,
        observation_space,
        architecture_name: str,
        architecture_config_path: str,
        entity_keys: Iterable[str] = ("humans",),
        entity_feature_dim: int = 14,
        mask_zero_entities: bool = True,
        include_waypoint_features: bool = False,
    ):
        robot_input_dim = effective_robot_input_dim(observation_space, include_waypoint_features)
        architecture_config = load_yaml(architecture_config_path)
        base_feature_dim = architecture_feature_dim(
            architecture_name,
            architecture_config,
            effective_robot_dim=robot_input_dim,
        )
        configured_robot_dim = int(architecture_config["model"]["robot_input_dim"])
        architecture_robot_input_dim = robot_input_dim if robot_input_dim > 0 and robot_input_dim != configured_robot_dim else None

        super().__init__(observation_space, features_dim=base_feature_dim)
        self.architecture_name = architecture_name
        self.base_robot_input_dim = _space_dim(observation_space, "robot")
        self.waypoint_input_dim = max(0, robot_input_dim - self.base_robot_input_dim)
        self.effective_robot_input_dim = robot_input_dim
        self.architecture = load_architecture(
            architecture_name,
            architecture_config_path,
            robot_input_dim=architecture_robot_input_dim,
        )
        self.entity_keys = tuple(entity_keys)
        self.entity_feature_dim = int(entity_feature_dim)
        self.mask_zero_entities = bool(mask_zero_entities)
        self.include_waypoint_features = bool(include_waypoint_features)

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        robot = observations["robot"].float()
        if self.include_waypoint_features and "waypoint_features" in observations:
            robot = self._extend_robot_observation(robot, observations["waypoint_features"].float())

        entities = _concat_entity_tensors(observations, self.entity_keys, self.entity_feature_dim)
        mask = _entity_mask(entities, self.mask_zero_entities)
        entity_history = entities if entities.dim() == 4 else entities.unsqueeze(2)
        result = self.architecture(robot, entity_history, mask, return_attention=True)
        if "features" in result:
            return result["features"]
        return torch.cat([result["robot_context"], result["social_context"]], dim=-1)

    def _extend_robot_observation(self, robot: torch.Tensor, waypoint_features: torch.Tensor) -> torch.Tensor:
        if robot.dim() == 2:
            waypoint_features = waypoint_features.reshape(robot.shape[0], -1)
            return torch.cat([robot, waypoint_features], dim=-1)
        if robot.dim() == 3:
            if waypoint_features.dim() != 3 or waypoint_features.shape[1] != robot.shape[1]:
                raise ValueError("Robot and waypoint histories must have the same number of timesteps.")
            return torch.cat([robot, waypoint_features], dim=-1)
        raise ValueError("Robot observation must be [batch, features] or [batch, time, features].")
