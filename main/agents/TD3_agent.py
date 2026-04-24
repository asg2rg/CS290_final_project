import torch
import numpy as np
from agents.actor import Actor
from agents.critic import Critic
from utils.replay import ReplayBuffer
import utils.configs as configs

class TD3Agent:
    def __init__(self, state_dim = 8, action_dim = 2, actor_lr=1e-3, critic_lr=1e-3):
        self.actor = Actor(state_dim, action_dim)
        self.critic_1 = Critic(state_dim, action_dim)
        self.critic_2 = Critic(state_dim, action_dim)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=critic_lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=critic_lr)
        # target networks
        self.actor_target = Actor(state_dim, action_dim)
        self.critic_1_target = Critic(state_dim, action_dim)
        self.critic_2_target = Critic(state_dim, action_dim)
        self._update_target_networks(tau=1.0)  # hard update at initialization
        self.replay_buffer = ReplayBuffer(capacity=1000000)

        # hyperparams
        self.gamma = 0.99
        self.tau = 0.005
        self.noise_std = 0.2
        self.noise_clip = 0.5

        self.target_speed = configs.TARGET_SPEED

        
    def _update_target_networks(self, tau=0.005):
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
        for target_param, param in zip(self.critic_1_target.parameters(), self.critic_1.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
        for target_param, param in zip(self.critic_2_target.parameters(), self.critic_2.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
    
    def make_decision(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)  # add batch dimension
        with torch.no_grad():
            action = self.actor(state_tensor).squeeze(0).numpy()  # remove batch dimension
        return action