from stable_baselines3 import PPO
# from gymnasium_env.envs.car_and_target import CarAndTargetEnv
from gymnasium_env.envs.env import CarAndTargetEnv

env = CarAndTargetEnv(render_mode="human", max_episode_steps=100)

# Load the saved model
model = PPO.load(r"C:\Users\JinHong\Desktop\simple_rl\tmp\best_model.zip")
# model = PPO.load("simple_rl")
obs, info = env.reset()

for _ in range(2000):
    action, _state = model.predict(obs, deterministic=True)
    print(action)
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        obs, info = env.reset()

env.close()