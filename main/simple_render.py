from gymnasium_env.envs.env import CarAndTargetEnv
import time
import utils.configs as configs

env = CarAndTargetEnv(render_mode="human", max_episode_steps=1000)
obs, info = env.reset()

for _ in range(1000):
    action = [0, 0]  
    # print("Action taken:", action)
    obs, reward, terminated, truncated, info = env.step(action)
    # print(obs, reward, terminated, truncated, info)

    if terminated or truncated:
        break

env.close()