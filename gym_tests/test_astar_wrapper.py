from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT))

import gym
import socnavgym

from global_planning.socnav_astar_wrapper import SocNavAStarWrapper


env = gym.make("SocNavGym-v1", config=str(ROOT / "env_configs" / "env_humans.yaml"))
env = SocNavAStarWrapper(
    env,
    config_path=str(ROOT / "global_planning" / "astar_wrapper_config.yaml"),
)
episode_seed = 1042
obs, info = env.reset(seed=episode_seed)
plan = env.plan_from_robot_to_goal()

print("Using SocNavGym from:", socnavgym.__file__)
print("Occupancy grid shape:", env.get_occupancy_grid().shape)
print("Path points:", len(plan.path_world))
print("Waypoints:", len(plan.waypoints))
print("Plan cost:", plan.cost)

for _ in range(10000):
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())

    if env.config["planner"]["replan_each_step"]:
        plan = env.latest_plan

    env.render()

    if terminated or truncated:
        episode_seed += 1
        obs, info = env.reset(seed=episode_seed)
        plan = env.plan_from_robot_to_goal()
        print("Length of waypoints: ", len(plan.waypoints))
