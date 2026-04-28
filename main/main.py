from gymnasium_env.envs.env import CarAndTargetEnv
import time
import utils.configs as configs
import numpy as np
import csv
from agents.TD3_agent import TD3Agent
import argparse

logs_dir = "logs"
ckpt_dir = "checkpoints"

def eval_loop(env, agent, prefix):
    configs.EVAL = True
    eval_log_path = "td3_eval_log.csv"
    # load checkpoints according to configs.DISCRETE and configs.CLAMP
    ckpt = "td3_checkpoint.pth"
    if configs.DISCRETE:
        raise NotImplementedError("TD3 with discrete action space is not implemented yet.")
    if configs.CLAMP:
        ckpt = "clamped_" + ckpt
        eval_log_path = "clamped_" + eval_log_path
    if prefix != "":
        ckpt = prefix + "_" + ckpt
        eval_log_path = prefix + "_" + eval_log_path
    
    # ckpt = ckpt_dir + "/" + ckpt
    eval_log_path = logs_dir + "/" + eval_log_path
    agent.load_checkpoint(ckpt)
    for ep in range(500): # run 500 evaluation episodes
        obs, info = env.reset()
        agent.init_hists()
        agent._add_obs_history(obs)
        episode_reward = 0.0
        done = False
        ep_step = 0

        while not done:
            obs_t = agent.parse_obs()
            action = agent.make_decision(obs_t, step=0) # no exploration noise during eval
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            agent.update_hists(next_obs, action)

            episode_reward += reward
            ep_step += 1

        print(f"Eval episode reward: {episode_reward}, steps: {ep_step}, distance traveled: {info['car_x'] - 100}")
        # add reward, steps, distance to eval log
        with open(eval_log_path, mode='a', newline='') as eval_log_file:
            eval_writer = csv.writer(eval_log_file)
            if eval_log_file.tell() == 0: # write header if file is new
                eval_writer.writerow(['episode', 'reward', 'steps', 'distance_traveled'])
            eval_writer.writerow([ep, episode_reward, ep_step, info['car_x'] - 100])


def main():
    save_path = "td3_checkpoint.pth"
    step_log_path = "td3_training_log.csv" # log losses over steps
    eps_log_path = "td3_episode_log.csv" #log rewards, dicounted rewards, and steps per episode
    eps_since_last_save = 0

    parser = argparse.ArgumentParser(description="Train TD3 agent in CarAndTargetEnv")
    parser.add_argument('--render', action='store_true', help='Render the environment')
    parser.add_argument('--save_path', type=str, default=save_path, help='Path to save the trained model checkpoint')
    parser.add_argument('--unclamp', action='store_true', default = False, help='Clamp actions to max values: only for continuous env')
    parser.add_argument('--discrete', action='store_true', default = False, help='Use discrete action space')
    parser.add_argument('--eval', action='store_true', default = False, help='Run evaluation loop after training')
    parser.add_argument('--exp-name', type=str, default="", help='Prefix for log and checkpoint files')
    args = parser.parse_args()
    
    print("##############################################")
    if args.render:
        configs.RENDER = True
    if not args.unclamp:
        configs.CLAMP = True
        save_path = "clamped_" + save_path
        step_log_path = "clamped_" + step_log_path
        eps_log_path = "clamped_" + eps_log_path
        print(f"Clamping outputs: actions are guaranteed to be in range{(-configs.MAX_ANG, configs.MAX_ANG), (-configs.MAX_ACC, configs.MAX_ACC)}")
    else:
        configs.CLAMP = False
        print("Not clamping outputs: actions can exceed max values, penalized by env.")
    if args.discrete:
        configs.DISCRETE = True
        print("Using discrete action space.")
    assert not (configs.DISCRETE and configs.CLAMP), "Cannot both clamp and use discrete spaces."
    if args.exp_name != "":
        save_path = args.exp_name + "_" + save_path
        step_log_path = args.exp_name + "_" + step_log_path
        eps_log_path = args.exp_name + "_" + eps_log_path
        print(f"Files will be saved to:\n\tCheckpoint: {save_path}\n\tStep log: {step_log_path}\n\tEpisode log: {eps_log_path}")
    if args.eval:
        configs.EVAL = True
        print("Eval mode set")
    print("##############################################")
    
    env = CarAndTargetEnv(render_mode="human" if configs.RENDER else None, max_episode_steps=500)
    agent = None
    if not configs.DISCRETE:
        agent = TD3Agent()
    else:
        raise NotImplementedError("TD3 with discrete action space is not implemented yet.")
    
    if configs.EVAL:
        eval_loop(env, agent, args.exp_name)
        return

    step = 0
    episode_cnt = 0
    try:    
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
                action = agent.make_decision(obs_t, step)
                r_obs = agent.build_replay_frame()
                # agent.decay_epsilon(step)
                next_obs, reward, terminated, truncated, info = env.step(action)
                agent.update_hists(next_obs, action)
                done = terminated or truncated
                r_n_obs = agent.build_replay_frame()
                agent.add_transition(r_obs, action, reward, r_n_obs, done) # internally advances histories

                # logging
                episode_reward += reward
                eps_disc_reward += (configs.DISCOUNT ** ep_step) * reward
                step += 1
                ep_step += 1

                # training loop
                loss_dict = agent.train(step)

                # log losses every 5000 steps
                if step % 2500 == 1 and loss_dict is not None:
                    with open(step_log_path, mode='a', newline='') as step_log_file:
                        step_writer = csv.writer(step_log_file)
                        if step_log_file.tell() == 0: # write header if file is new
                            step_writer.writerow(['step', 'actor_loss', 'critic_1_loss', 'critic_2_loss', 'noise_std'])
                        step_writer.writerow([step, loss_dict['actor_loss'], loss_dict['critic_1_loss'], loss_dict['critic_2_loss'], loss_dict['noise_std']])

            if episode_cnt % 10 == 0:
                print(f"Episode {episode_cnt} reward: {episode_reward}")
                # log episode reward and discounted reward
                dist_traveled = info["car_x"] - 100
                with open(eps_log_path, mode='a', newline='') as eps_log_file:
                    eps_writer = csv.writer(eps_log_file)
                    if eps_log_file.tell() == 0: # write header if file is new
                        eps_writer.writerow(['episode', 'steps', 'reward', 'discounted_reward', 'steps', 'distance_traveled'])
                    eps_writer.writerow([episode_cnt, step, episode_reward, eps_disc_reward, ep_step, dist_traveled])
            episode_cnt += 1
            eps_since_last_save += 1
            if ep_step == env.max_episode_steps:
                if eps_since_last_save >= 30:
                    agent.save_checkpoint(save_path)
                    print(f"Checkpoint saved to {save_path}.")
                    eps_since_last_save = 0

        env.close()
        agent.save_checkpoint(save_path)
    except KeyboardInterrupt:
        print("Training interrupted. Saving checkpoint...")
        agent.save_checkpoint(save_path)
        print(f"Checkpoint saved to {save_path}. Exiting.")
        env.close()

if __name__ == "__main__":
    main()