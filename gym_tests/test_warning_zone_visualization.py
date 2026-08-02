from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT))

import gym
import socnavgym

from custom_rewards import WarningZoneVisualizationWrapper


env = gym.make("SocNavGym-v1", config=str(ROOT / "env_configs" / "env_main.yaml"))
env = WarningZoneVisualizationWrapper(
    env,
    config_path=str(ROOT / "custom_rewards" / "warning_zone_visualization_config.yaml"),
)

obs, info = env.reset(seed=6249)

print("Using SocNavGym from:", socnavgym.__file__)
print("Warning-zone visualization enabled:", env.config["visualization"]["enabled"])
print("Render callbacks:", len(env.unwrapped.render_callbacks))

for _ in range(1000):
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    env.render()

    if terminated or truncated:
        obs, info = env.reset()
