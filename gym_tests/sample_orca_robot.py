import sys
from pathlib import Path

import gym
import socnavgym  # noqa: F401 - registers SocNavGym-v1 with gym

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from testing_pipeline.policies import ORCARobotPolicy


env = gym.make("SocNavGym-v1", config="./env_configs/env_main.yaml")
policy = ORCARobotPolicy()

episode_seed = 42
obs, _ = env.reset(seed=episode_seed)

for _ in range(1000):
    action = policy.predict(obs, env)
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()

    if terminated or truncated:
        episode_seed += 1
        obs, _ = env.reset(seed=episode_seed)
