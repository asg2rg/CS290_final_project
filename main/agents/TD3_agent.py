import torch
import numpy as np
from agents.actor import Actor
from agents.critic import Critic
from utils.replay import ReplayBuffer
import utils.configs as configs

class TD3Agent:
    def __init__(self, state_dim = 8, action_dim = 2, actor_lr=1e-3, critic_lr=1e-3):
        self.stack_sz = configs.STACK_SZ
        self.state_dim = state_dim
        self.act_dim = action_dim
        # obs size: target speed + current obs + past (obs, action) pairs
        self.obs_dims = 1 + self.state_dim * self.stack_sz + self.act_dim * (self.stack_sz - 1)
        if torch.cuda.is_available():
            print("Using GPU")
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            print("Using Apple Silicon GPU")
            self.device = torch.device("mps")
        else:
            print("Using CPU")
            self.device = torch.device("cpu")
        self.actor = Actor(self.obs_dims, action_dim).to(self.device)
        self.critic_1 = Critic(self.obs_dims, action_dim).to(self.device)
        self.critic_2 = Critic(self.obs_dims, action_dim).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=critic_lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=critic_lr)
        # target networks
        self.actor_target = Actor(self.obs_dims, action_dim).to(self.device)
        self.critic_1_target = Critic(self.obs_dims, action_dim).to(self.device)
        self.critic_2_target = Critic(self.obs_dims, action_dim).to(self.device)
        self._update_target_networks(tau=1.0)  # hard update at initialization
        self.replay_buffer = ReplayBuffer(capacity=1000000)

        # histories
        self.obs_history = [] # list of np arrays, each of shape (obs_dim,)
        self.act_history = [] # list of np arrays, each of shape (action_dim,)

        # hyperparams
        self.gamma = 0.99
        self.tau = 0.005
        self.noise_std = 0.2
        self.noise_clip = 0.5

        self.epsilon = configs.EPS_START
        self.epsilon_min = configs.EPS_MIN
        self.epsilon_decay = configs.EPS_DECAY
        self.decay_interval = configs.DECAY_INTERVAL

        self.target_speed = configs.TARGET_SPEED
    
    def init_hists(self):
        self.obs_history = []
        self.act_history = []
        # fill history with zeros
        for _ in range(self.stack_sz):
            self.obs_history.append(np.zeros(self.state_dim, dtype=np.float32))
            self.act_history.append(np.zeros(self.act_dim, dtype=np.float32))
        self.act_history.pop(0)
    
    def _add_obs_history(self, obs):
        self.obs_history.append(obs)
        if len(self.obs_history) > self.stack_sz:
            self.obs_history.pop(0)
    
    def _add_act_history(self, action):
        self.act_history.append(action)
        if len(self.act_history) > self.stack_sz-1:
            self.act_history.pop(0)
        
    def parse_obs(self):
        parts = [np.array([self.target_speed], dtype=np.float32)] + self.obs_history + self.act_history
        obs_np = np.concatenate(parts, axis=0)
        obs_t = torch.FloatTensor(obs_np).unsqueeze(0).to(self.device)
        return obs_t

    def add_transition(self, action, reward, next_obs, done):
        # print(f"Shape of obs history: {[h.shape for h in self.obs_history]}")
        # print(f"Shape of act history: {[h.shape for h in self.act_history]}")
        replay_parts = [np.array([self.target_speed], dtype=np.float32)] + self.obs_history + self.act_history
        replay_obs = np.concatenate(replay_parts, axis=0)
        # print(f"Shape of replay obs: {replay_obs.shape}")
        self._add_obs_history(next_obs)
        self._add_act_history(action)
        # print(f"Shape of updated obs history: {[h.shape for h in self.obs_history]}")
        # print(f"Shape of updated act history: {[h.shape for h in self.act_history]}")
        replay_next_parts = [np.array([self.target_speed], dtype=np.float32)] + self.obs_history + self.act_history
        replay_next = np.concatenate(replay_next_parts, axis=0)
        # print(f"Shape of replay next: {replay_next.shape}")
        self.replay_buffer.add(replay_obs, action, reward, replay_next, done)
        
    def _update_target_networks(self, tau=0.005):
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
        for target_param, param in zip(self.critic_1_target.parameters(), self.critic_1.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
        for target_param, param in zip(self.critic_2_target.parameters(), self.critic_2.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
    
    def make_decision(self, obs_t):
        # print(f"Processing obs[{obs_t.shape}]: {obs_t.cpu().numpy()}")
        greed = np.random.rand()
        if greed < self.epsilon:
            # explore: random action
            action = np.random.uniform(low=[-configs.MAX_ANG, -configs.MAX_ACC], high=[configs.MAX_ANG, configs.MAX_ACC], size=(self.act_dim,))
        else:
            # exploit: action from actor
            with torch.no_grad():
                action = self.actor(obs_t).squeeze(0).cpu().numpy()
        return action

    def train(self, step):
        if len(self.replay_buffer) < configs.BATCH_SIZE:
            return
        
        batch_obs, batch_actions, batch_rewards, batch_next_obs, batch_dones = self.replay_buffer.sample(configs.BATCH_SIZE, device=self.device)
        # compute target Q value
        with torch.no_grad():
            noise = (torch.randn_like(batch_actions) * self.noise_std).clamp(-self.noise_clip, self.noise_clip).to(self.device)
            next_actions = (self.actor_target(batch_next_obs) + noise).clamp(-configs.MAX_ANG, configs.MAX_ACC)
            target_q1 = self.critic_1_target(batch_next_obs, next_actions)
            target_q2 = self.critic_2_target(batch_next_obs, next_actions)
            target_q = torch.min(target_q1, target_q2)
            target_q = batch_rewards + (1 - batch_dones) * self.gamma * target_q
        
        # update critic networks
        current_q1 = self.critic_1(batch_obs, batch_actions)
        current_q2 = self.critic_2(batch_obs, batch_actions)
        critic_1_loss = torch.nn.MSELoss()(current_q1, target_q)
        critic_2_loss = torch.nn.MSELoss()(current_q2, target_q)

        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()

        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        # delayed policy updates
        if step % 2 == 1:
            # update actor network
            actor_loss = -self.critic_1(batch_obs, self.actor(batch_obs)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # update target networks
            self._update_target_networks(tau=self.tau)

            # print every 500 steps
            if step % 500 == 1:
                print(f"Step {step}: Actor loss = {actor_loss.item():.2f}, Critic 1 loss = {critic_1_loss.item():.2f}, Critic 2 loss = {critic_2_loss.item():.2f}, Epsilon = {self.epsilon:.2f}")

    def decay_epsilon(self, step):
        if step % self.decay_interval == 0 and self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon, self.epsilon_min)