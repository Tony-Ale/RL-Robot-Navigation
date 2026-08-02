from typing import Any

import numpy as np


class LearnedAgentPolicy:
    """Adapter around a trained Stable-Baselines3 model."""

    controller_name = "learned_agent"

    def __init__(self, model: Any, deterministic: bool = True):
        self.model = model
        self.deterministic = bool(deterministic)

    def reset(self) -> None:
        """Reset episode-local policy state; ordinary PPO has none."""

    def predict(self, observation, env=None):
        action, _ = self.model.predict(observation, deterministic=self.deterministic)
        return action


class ORCARobotPolicy:
    """Drive the active robot with SocNavGym's robot-side ORCA velocity logic."""

    controller_name = "orca"

    def reset(self) -> None:
        """Reset episode-local policy state; SocNavGym owns ORCA state."""

    def predict(self, observation, env):
        base_env = find_env_with_attr(env, "compute_orca_velocity_robot")
        if base_env is None:
            raise AttributeError("Could not find SocNavGym env with compute_orca_velocity_robot for ORCA baseline.")

        robot = getattr(base_env, "robot", None)
        if robot is None:
            raise AttributeError("Could not find active robot on SocNavGym env for ORCA baseline.")

        velocity = base_env.compute_orca_velocity_robot(robot)
        return orca_velocity_to_action(base_env, robot, velocity, env.action_space.shape)


def find_env_with_attr(env, attr: str):
    """Walk through Gym/Gymnasium wrappers until an env exposing attr is found."""
    current = env
    visited = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        if hasattr(current, attr):
            return current
        current = getattr(current, "env", None)
    unwrapped = getattr(env, "unwrapped", None)
    if unwrapped is not None and hasattr(unwrapped, attr):
        return unwrapped
    return None


def orca_velocity_to_action(base_env, robot, velocity, action_shape):
    """Convert SocNavGym ORCA world velocity into the normalized action interface."""
    velocity = np.asarray(velocity, dtype=np.float32)
    max_advance = max(float(getattr(base_env, "MAX_ADVANCE_ROBOT")), 1e-8)
    max_rotation = max(float(getattr(base_env, "MAX_ROTATION")), 1e-8)
    timestep = max(float(getattr(base_env, "TIMESTEP")), 1e-8)

    if getattr(robot, "type", "diff-drive") == "holonomic":
        linear_x = velocity[0] * np.cos(robot.orientation) + velocity[1] * np.sin(robot.orientation)
        linear_y = -velocity[0] * np.sin(robot.orientation) + velocity[1] * np.cos(robot.orientation)
        angular = (np.arctan2(velocity[1], velocity[0]) - robot.orientation) / timestep
    else:
        linear_x = np.sqrt(float(velocity[0]) ** 2 + float(velocity[1]) ** 2)
        linear_y = 0.0
        angular = (np.arctan2(velocity[1], velocity[0]) - robot.orientation) / timestep

    normalized = np.array(
        [
            np.clip(linear_x / max_advance, -1.0, 1.0),
            np.clip(linear_y / max_advance, -1.0, 1.0),
            np.clip(angular / max_rotation, -1.0, 1.0),
        ],
        dtype=np.float32,
    )
    if len(action_shape) == 1 and action_shape[0] == 2:
        return np.array([normalized[0], normalized[2]], dtype=np.float32)
    return normalized
