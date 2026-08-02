import socnavgym
import gym
env = gym.make("SocNavGym-v1", config="./env_configs/env_main.yaml")
episode_seed = 1042
obs, _ = env.reset(seed=episode_seed)


for i in range(1000):
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    env.render()
    if terminated or truncated:
        episode_seed += 1
        env.reset(seed=episode_seed)
