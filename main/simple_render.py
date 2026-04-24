from gymnasium_env.envs.env import CarAndTargetEnv
import time

env = CarAndTargetEnv(render_mode="human", max_episode_steps=1000)
obs, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()  # random action for testing
    # print("Action taken:", action)
    obs, reward, terminated, truncated, info = env.step(action)
    # print(obs, reward, terminated, truncated, info)

    if terminated or truncated:
        break

env.close()