from gymnasium_env.envs.env import CarAndTargetEnv
import time
import utils.configs as configs
import numpy as np
import torch
from agents.TD3_agent import TD3Agent

env = CarAndTargetEnv(render_mode="human" if configs.RENDER else None, max_episode_steps=1000)
agent = TD3Agent()

step = 0
episode_cnt = 0
while step < (configs.G_STEPS):
    obs, info = env.reset()
    agent.init_hists()
    agent._add_obs_history(obs)
    episode_reward = 0.0
    done = False

    while not done:
        # interaction loop
        obs_t = agent.parse_obs()
        action = agent.make_decision(obs_t)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        agent.add_transition(action, reward, next_obs, done) # internally advances histories

        # logging
        episode_reward += reward
        step += 1

        # training loop
        agent.train(step)
    
    if episode_cnt % 500 == 0:
        print(f"Episode {episode_cnt} reward: {episode_reward}")
    episode_cnt += 1

env.close()