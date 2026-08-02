import torch
from gymnasium import spaces

from sb3_contrib.common.recurrent.buffers import RecurrentDictRolloutBuffer, RecurrentRolloutBuffer
from sb3_contrib.common.recurrent.type_aliases import RNNStates
from sb3_contrib.ppo_recurrent import RecurrentPPO

from stateful_training_pipeline.policy import StatefulSocialContextPolicy


class StatefulSocialRecurrentPPO(RecurrentPPO):
    """RecurrentPPO using packed robot and per-entity GRU states."""

    policy_aliases = {"StatefulSocialContextPolicy": StatefulSocialContextPolicy}

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)
        buffer_class = RecurrentDictRolloutBuffer if isinstance(self.observation_space, spaces.Dict) else RecurrentRolloutBuffer

        self.policy = self.policy_class(
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            use_sde=self.use_sde,
            **self.policy_kwargs,
        ).to(self.device)

        state_size = self.policy.lstm_actor.hidden_size
        state_shape = (1, self.n_envs, state_size)
        zeros = lambda: torch.zeros(state_shape, device=self.device)
        self._last_lstm_states = RNNStates((zeros(), zeros()), (zeros(), zeros()))
        buffer_state_shape = (self.n_steps, 1, self.n_envs, state_size)
        self.rollout_buffer = buffer_class(
            self.n_steps,
            self.observation_space,
            self.action_space,
            buffer_state_shape,
            self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
        )

        from stable_baselines3.common.utils import get_schedule_fn

        self.clip_range = get_schedule_fn(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)) and self.clip_range_vf <= 0:
                raise ValueError("clip_range_vf must be positive or null.")
            self.clip_range_vf = get_schedule_fn(self.clip_range_vf)
