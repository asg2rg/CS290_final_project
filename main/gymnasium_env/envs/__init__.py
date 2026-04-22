from gymnasium_env.envs.env import CarAndTargetEnv

from gymnasium.envs.registration import register



register(
    id="gymnasium_env/CarAndTarget-v0",
    entry_point="gymnasium_env.envs:CarAndTargetEnv"
)

