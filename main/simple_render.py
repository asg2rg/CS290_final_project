from gymnasium_env.envs.env import CarAndTargetEnv
import time

env = CarAndTargetEnv(render_mode="human", max_episode_steps=100)
obs, info = env.reset()

time.sleep(15)   # keep window open for 5 seconds
env.close()