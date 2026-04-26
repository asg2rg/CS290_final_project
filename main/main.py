from gymnasium_env.envs.env import CarAndTargetEnv
import time
import utils.configs as configs
import numpy as np
import csv
from agents.TD3_agent import TD3Agent

env = CarAndTargetEnv(render_mode="human" if configs.RENDER else None, max_episode_steps=1000)
agent = TD3Agent()

save_path = "td3_checkpoint.pth"
step_log_path = "td3_training_log.csv" # log losses over steps
eps_log_path = "td3_episode_log.csv" #log rewards, dicounted rewards, and steps per episode

step = 0
episode_cnt = 0
while step < (configs.G_STEPS):
    obs, info = env.reset()
    agent.init_hists()
    agent._add_obs_history(obs)
    episode_reward = 0.0
    eps_disc_reward = 0.0
    done = False
    ep_step = 0

    while not done:
        # interaction loop
        obs_t = agent.parse_obs()
        action = agent.make_decision(obs_t)
        agent.decay_epsilon(step)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        agent.add_transition(action, reward, next_obs, done) # internally advances histories

        # logging
        episode_reward += reward
        eps_disc_reward += (configs.DISCOUNT ** ep_step) * reward
        step += 1
        ep_step += 1

        # training loop
        loss_dict = agent.train(step)

        # log losses every 1000 steps
        if step % 1000 == 1 and loss_dict is not None:
            with open(step_log_path, mode='a', newline='') as step_log_file:
                step_writer = csv.writer(step_log_file)
                if step_log_file.tell() == 0: # write header if file is new
                    step_writer.writerow(['step', 'actor_loss', 'critic_1_loss', 'critic_2_loss', 'epsilon'])
                step_writer.writerow([step, loss_dict['actor_loss'], loss_dict['critic_1_loss'], loss_dict['critic_2_loss'], loss_dict['epsilon']])
    
    # log episode reward and discounted reward
    dist_traveled = info["car_x"] - 100
    with open(eps_log_path, mode='a', newline='') as eps_log_file:
        eps_writer = csv.writer(eps_log_file)
        if eps_log_file.tell() == 0: # write header if file is new
            eps_writer.writerow(['episode', 'reward', 'discounted_reward', 'steps', 'distance_traveled'])
        eps_writer.writerow([episode_cnt, episode_reward, eps_disc_reward, ep_step, dist_traveled])

    if episode_cnt % 500 == 0:
        print(f"Episode {episode_cnt} reward: {episode_reward}")
    episode_cnt += 1

env.close()
agent.save_checkpoint(save_path)