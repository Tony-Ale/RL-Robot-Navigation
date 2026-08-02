from pathlib import Path
from typing import Optional

import numpy as np

from stateful_training_pipeline.recurrent_ppo import StatefulSocialRecurrentPPO


class StatefulLearnedAgentPolicy:
    """Carry recurrent model state and reset it with each SocNavGym episode."""

    controller_name = "learned_agent"

    def __init__(self, model: StatefulSocialRecurrentPPO, deterministic: bool = True):
        self.model = model
        self.deterministic = bool(deterministic)
        self.reset()

    def reset(self) -> None:
        self.state = None
        self.episode_start = np.ones((1,), dtype=bool)
        self._last_tick: Optional[int] = None

    def predict(self, observation, env=None):
        tick = self._environment_tick(env)
        if tick == 0 and self._last_tick != 0:
            self.reset()
        action, self.state = self.model.predict(
            observation,
            state=self.state,
            episode_start=self.episode_start,
            deterministic=self.deterministic,
        )
        self.episode_start[:] = False
        self._last_tick = tick
        return action

    @staticmethod
    def _environment_tick(env) -> Optional[int]:
        if env is None:
            return None
        base_env = getattr(env, "unwrapped", None)
        tick = getattr(base_env, "ticks", None)
        return None if tick is None else int(tick)


def load_stateful_policy(
    checkpoint: Path,
    env=None,
    deterministic: bool = True,
    device: str = "auto",
) -> StatefulLearnedAgentPolicy:
    model = StatefulSocialRecurrentPPO.load(str(checkpoint), env=env, device=device)
    model.policy.set_training_mode(False)
    return StatefulLearnedAgentPolicy(model, deterministic=deterministic)
