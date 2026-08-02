from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT))

import gym
import socnavgym

from global_planning.socnav_astar_wrapper import SocNavAStarWrapper
from navigation_features import CoordinateFrameWaypointWrapper


env = gym.make("SocNavGym-v1", config=str(ROOT / "env_configs" / "env_humans.yaml"))
env = SocNavAStarWrapper(
    env,
    config_path=str(ROOT / "global_planning" / "astar_wrapper_config.yaml"),
)
env = CoordinateFrameWaypointWrapper(
    env,
    config_path=str(ROOT / "navigation_features" / "config.yaml"),
)

obs, info = env.reset(seed=624)
waypoint_key = env.config["waypoint_features"]["observation_key"]

print("Using SocNavGym from:", socnavgym.__file__)
print("Coordinate frame:", env.config["coordinate_frame"]["mode"])
print("Waypoint feature shape:", obs[waypoint_key].shape)
print("Waypoint observation-space shape:", env.observation_space[waypoint_key].shape)
print("Robot goal in active frame:", obs["robot"][6:8])
print("First waypoint feature row:", obs[waypoint_key].reshape(-1, 4)[0])
