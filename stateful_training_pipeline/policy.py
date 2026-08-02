from typing import Any, Dict, Iterable, List, Optional, Tuple, Type, Union

import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.distributions import Distribution
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, MlpExtractor
from stable_baselines3.common.type_aliases import Schedule
from torch import nn

from architectures.stateful_social_context_fusion import StatefulSocialContextFusionNet, load_architecture_config
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from sb3_contrib.common.recurrent.type_aliases import RNNStates


class _FeatureShapeExtractor(BaseFeaturesExtractor):
    """Declare the recurrent architecture output size to SB3's policy heads."""

    def __init__(self, observation_space, features_dim: int):
        super().__init__(observation_space, features_dim)

    def forward(self, observations):
        raise RuntimeError("StatefulSocialContextPolicy processes dictionary observations directly.")


class _PackedStateSpec(nn.Module):
    """Expose the packed GRU state shape expected by sb3-contrib's buffer."""

    num_layers = 1

    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = int(hidden_size)


class StatefulSocialContextPolicy(RecurrentActorCriticPolicy):
    """Recurrent PPO policy backed by robot and per-entity stateful GRUs."""

    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        architecture_config_path: str,
        entity_keys: Iterable[str] = ("humans",),
        entity_feature_dim: int = 14,
        mask_zero_entities: bool = True,
        include_waypoint_features: bool = False,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        **kwargs: Any,
    ):
        if not isinstance(observation_space, spaces.Dict):
            raise ValueError("StatefulSocialContextPolicy requires a dictionary observation space.")

        self.entity_keys = tuple(entity_keys)
        self.entity_feature_dim = int(entity_feature_dim)
        self.mask_zero_entities = bool(mask_zero_entities)
        self.include_waypoint_features = bool(include_waypoint_features)
        self.architecture_config_path = str(architecture_config_path)
        robot_input_dim = self._robot_input_dim(observation_space)
        num_entities = self._entity_count(observation_space)
        architecture_config = load_architecture_config(architecture_config_path)
        configured_entity_dim = int(architecture_config["model"]["entity_input_dim"])
        if configured_entity_dim != self.entity_feature_dim:
            raise ValueError(
                "architecture.entity_feature_dim must match model.entity_input_dim: "
                f"{self.entity_feature_dim} != {configured_entity_dim}."
            )
        architecture = StatefulSocialContextFusionNet(architecture_config, robot_input_dim, num_entities)
        self._recurrent_output_dim = architecture.output_dim
        packed_state_size = architecture.state_size

        ActorCriticPolicy.__init__(
            self,
            observation_space,
            action_space,
            lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            ortho_init=ortho_init,
            use_sde=use_sde,
            features_extractor_class=_FeatureShapeExtractor,
            features_extractor_kwargs={"features_dim": architecture.output_dim},
            **kwargs,
        )

        # sb3-contrib stores two LSTM tensors. The first carries the packed GRU
        # state; the second remains zero so its standard recurrent buffer can be reused.
        self.lstm_actor = _PackedStateSpec(packed_state_size)
        self.lstm_critic = None
        self.critic = None
        self.lstm_output_dim = architecture.output_dim
        self.lstm_kwargs = {}
        self.shared_lstm = False
        self.enable_critic_lstm = False
        self.architecture = architecture
        self.lstm_hidden_state_shape = (1, 1, packed_state_size)
        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.pop("features_extractor_class", None)
        data.pop("features_extractor_kwargs", None)
        data.update(
            architecture_config_path=self.architecture_config_path,
            entity_keys=self.entity_keys,
            entity_feature_dim=self.entity_feature_dim,
            mask_zero_entities=self.mask_zero_entities,
            include_waypoint_features=self.include_waypoint_features,
        )
        return data

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = MlpExtractor(
            self._recurrent_output_dim,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            device=self.device,
        )

    def forward(
        self,
        obs: Dict[str, torch.Tensor],
        states: RNNStates,
        episode_starts: torch.Tensor,
        deterministic: bool = False,
    ):
        policy_features, policy_state = self._process_observations(obs, states.pi, episode_starts)
        latent_policy = self.mlp_extractor.forward_actor(policy_features)
        latent_value = self.mlp_extractor.forward_critic(policy_features)
        values = self.value_net(latent_value)
        distribution = self._get_action_dist_from_latent(latent_policy)
        actions = distribution.get_actions(deterministic=deterministic)
        return actions, values, distribution.log_prob(actions), RNNStates(policy_state, policy_state)

    def evaluate_actions(
        self,
        obs: Dict[str, torch.Tensor],
        actions: torch.Tensor,
        states: RNNStates,
        episode_starts: torch.Tensor,
    ):
        policy_features, _ = self._process_observations(obs, states.pi, episode_starts)
        latent_policy = self.mlp_extractor.forward_actor(policy_features)
        latent_value = self.mlp_extractor.forward_critic(policy_features)
        distribution = self._get_action_dist_from_latent(latent_policy)
        return self.value_net(latent_value), distribution.log_prob(actions), distribution.entropy()

    def get_distribution(
        self,
        obs: Dict[str, torch.Tensor],
        states: Tuple[torch.Tensor, torch.Tensor],
        episode_starts: torch.Tensor,
    ) -> Tuple[Distribution, Tuple[torch.Tensor, torch.Tensor]]:
        features, new_state = self._process_observations(obs, states, episode_starts)
        latent = self.mlp_extractor.forward_actor(features)
        return self._get_action_dist_from_latent(latent), new_state

    def predict_values(
        self,
        obs: Dict[str, torch.Tensor],
        states: Tuple[torch.Tensor, torch.Tensor],
        episode_starts: torch.Tensor,
    ) -> torch.Tensor:
        features, _ = self._process_observations(obs, states, episode_starts)
        return self.value_net(self.mlp_extractor.forward_critic(features))

    def _predict(
        self,
        observation: Dict[str, torch.Tensor],
        lstm_states: Tuple[torch.Tensor, torch.Tensor],
        episode_starts: torch.Tensor,
        deterministic: bool = False,
    ):
        distribution, new_state = self.get_distribution(observation, lstm_states, episode_starts)
        return distribution.get_actions(deterministic=deterministic), new_state

    def _process_observations(
        self,
        observations: Dict[str, torch.Tensor],
        states: Tuple[torch.Tensor, torch.Tensor],
        episode_starts: torch.Tensor,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        robot = observations["robot"].float().reshape(observations["robot"].shape[0], -1)
        if self.include_waypoint_features:
            if "waypoint_features" not in observations:
                raise ValueError("include_waypoint_features requires the waypoint_features observation key.")
            waypoint = observations["waypoint_features"].float().reshape(robot.shape[0], -1)
            robot = torch.cat((robot, waypoint), dim=-1)

        entity_batches = []
        for key in self.entity_keys:
            if key not in observations:
                raise ValueError(f"Configured entity key '{key}' is missing from observations.")
            values = observations[key].float()
            entity_batches.append(values.reshape(values.shape[0], -1, self.entity_feature_dim))
        entities = torch.cat(entity_batches, dim=1)
        if self.mask_zero_entities:
            mask = torch.sum(torch.abs(entities), dim=-1) > 1e-8
        else:
            mask = torch.ones(entities.shape[:2], dtype=torch.bool, device=entities.device)

        features, packed_state = self.architecture.forward_sequence(
            robot, entities, mask, states[0], episode_starts
        )
        return features, (packed_state, torch.zeros_like(packed_state))

    def _robot_input_dim(self, observation_space: spaces.Dict) -> int:
        if "robot" not in observation_space.spaces:
            raise ValueError("The observation space must contain robot features.")
        dimension = int(np.prod(observation_space["robot"].shape))
        if self.include_waypoint_features:
            if "waypoint_features" not in observation_space.spaces:
                raise ValueError("include_waypoint_features requires waypoint_features in the observation space.")
            dimension += int(np.prod(observation_space["waypoint_features"].shape))
        return dimension

    def _entity_count(self, observation_space: spaces.Dict) -> int:
        count = 0
        for key in self.entity_keys:
            if key not in observation_space.spaces:
                raise ValueError(f"Configured entity key '{key}' is missing from the observation space.")
            size = int(np.prod(observation_space[key].shape))
            if size % self.entity_feature_dim != 0:
                raise ValueError(
                    f"Observation key '{key}' has {size} values, which is not divisible by "
                    f"entity_feature_dim={self.entity_feature_dim}."
                )
            count += size // self.entity_feature_dim
        return count
