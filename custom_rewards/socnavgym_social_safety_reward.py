import math
from pathlib import Path

from socnavgym.envs.rewards.reward_api import RewardAPI

from custom_rewards.social_safety_reward import (
    compute_social_safety_reward,
    load_social_safety_reward_config,
    reset_stagnation_tracking,
)


class Reward(RewardAPI):
    """SocNavGym custom reward adapter for the project social-safety reward."""

    def __init__(self, env):
        super().__init__(env)
        config_path = Path(__file__).with_name("social_safety_reward_config.yaml")
        self.config = load_social_safety_reward_config(config_path)
        self.previous_goal_distance = self._goal_distance()
        reset_stagnation_tracking(self.env)

    def compute_reward(self, action, prev_obs, curr_obs):
        previous_goal_distance = self.previous_goal_distance
        if previous_goal_distance is None:
            previous_goal_distance = self._goal_distance()

        reward, reward_info = compute_social_safety_reward(
            self.env,
            previous_goal_distance=previous_goal_distance,
            config=self.config,
            reached_goal=self.check_reached_goal(),
            timeout=self.check_timeout(),
            collision=self.check_collision(),
        )
        self.previous_goal_distance = reward_info["goal_distance"]
        self._update_info(reward, reward_info)
        return reward

    def _goal_distance(self):
        robot = self.env.robot
        if robot is None:
            return None
        return math.hypot(robot.x - robot.goal_x, robot.y - robot.goal_y)

    def _update_info(self, reward, reward_info):
        self.info["DISCOMFORT_SNGNN"] = 0.0
        self.info["DISCOMFORT_DSRNN"] = reward_info["warning_zone_reward"]
        if reward_info["reward_reason"] == "shaped":
            self.info["distance_reward"] = (
                reward
                - reward_info["warning_zone_reward"]
                - reward_info["static_warning_zone_reward"]
                - reward_info["checkpoint_reward"]
                - reward_info["stagnation_penalty"]
            )
        else:
            self.info["distance_reward"] = 0.0
        self.info["alive_reward"] = 0.0
        self.info["sngnn_reward"] = 0.0
        self.info["custom_reward"] = reward
        for key, value in reward_info.items():
            self.info[key] = value
