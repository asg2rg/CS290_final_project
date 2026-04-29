import torch
import numpy as np
from agents.actor import Actor
from agents.critic import Critic
from utils.replay import ReplayBuffer
import utils.configs as configs

class TD3Agent:
    def __init__(self, state_dim = 8, action_dim = 2, actor_lr=1e-4, critic_lr=1e-4):
        self.stack_sz = configs.STACK_SZ
        self.state_dim = state_dim
        self.act_dim = action_dim
        self.obs_dims = configs.STACK_OBS_DIM
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
        self.replay_buffer = ReplayBuffer(capacity=500000)

        self.action_low = np.array([-configs.MAX_ANG, -configs.MAX_ACC], dtype=np.float32)
        self.action_high = np.array([configs.MAX_ANG, configs.MAX_ACC], dtype=np.float32)
        self.action_scale = np.array([configs.MAX_ANG, configs.MAX_ACC], dtype=np.float32)

        # histories
        self.obs_history = [] # list of np arrays, each of shape (obs_dim,)
        self.act_history = [] # list of np arrays, each of shape (action_dim,)

        # hyperparams
        self.gamma = configs.DISCOUNT
        self.tau = 0.005
        self.noise_std = 0.2
        self.noise_clip = 0.5
        self.grad_clip = 5.0

        self.epsilon = configs.EPS_START
        self.epsilon_min = configs.EPS_MIN
        self.epsilon_decay = configs.EPS_DECAY
        self.decay_interval = configs.DECAY_INTERVAL
        self.explore_noise_std = configs.EXPLORE_NOISE_STD
        self.explore_noise_min = configs.EXPLORE_NOISE_MIN
    
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
    
    def update_hists(self, obs, action):
        self._add_obs_history(obs)
        self._add_act_history(action)

    def build_replay_frame(self):
        parts = [np.array([configs.TARGET_SPEED, configs.TARGET_LANE], dtype=np.float32)] + self.obs_history + self.act_history
        obs_np = np.concatenate(parts, axis=0)
        return obs_np

    def parse_obs(self):
        parts = [np.array([configs.TARGET_SPEED, configs.TARGET_LANE], dtype=np.float32)] + self.obs_history + self.act_history
        
        obs_np = np.concatenate(parts, axis=0)
        obs_t = torch.FloatTensor(obs_np).unsqueeze(0).to(self.device)
        return obs_t

    def add_transition(self, obs_t, action, reward, next_obs_t, done):
        self.replay_buffer.add(obs_t, action, reward, next_obs_t, done)
        
    def _update_target_networks(self, tau=0.005):
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
        for target_param, param in zip(self.critic_1_target.parameters(), self.critic_1.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
        for target_param, param in zip(self.critic_2_target.parameters(), self.critic_2.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
    
    def make_decision(self, obs_t, step):
        if configs.EVAL:
            with torch.no_grad():
                action = self.actor(obs_t).squeeze(0).cpu().numpy()
            return action
        # print(f"Processing obs[{obs_t.shape}]: {obs_t.cpu().numpy()}")
        # greed = np.random.rand()
        # if greed < self.epsilon:
            # explore: random action
        if not configs.DISCRETE:
            if step < configs.G_STEPS*0.05:
                action = np.random.uniform(low=self.action_low, high=self.action_high, size=(self.act_dim,)).astype(np.float32)
            else:
                # exploit: action from actor
                with torch.no_grad():
                    action = self.actor(obs_t).squeeze(0).cpu().numpy()
                # exploration noise
                noise_std = self.get_noise_std(step) * self.action_scale
                noise = np.random.normal(0.0, noise_std, size=action.shape).astype(np.float32)
                if configs.CLAMP:
                    action = (action + noise).clip(self.action_low, self.action_high)
                else:
                    action = action + noise
            return action
        else:
            if np.random.rand() < self.epsilon:
                action = np.random.randint(0, configs.TURN_SHAPE, size=(1,)).tolist() + np.random.randint(0, configs.ACC_SHAPE, size=(1,)).tolist()
            else:
                with torch.no_grad():
                    action = self.actor(obs_t).squeeze(0).cpu().numpy()
                ... # TODO

    def train(self, step):
        if len(self.replay_buffer) < configs.BATCH_SIZE:
            return
        
        loss_dict = {}
        batch_obs, batch_actions, batch_rewards, batch_next_obs, batch_dones = self.replay_buffer.sample(configs.BATCH_SIZE, device=self.device)
        # compute target Q value
        with torch.no_grad():
            target_noise_std = torch.tensor(self.action_scale * self.noise_std, device=self.device)
            target_noise_clip = torch.tensor(self.action_scale * self.noise_clip, device=self.device)
            noise = torch.randn_like(batch_actions) * target_noise_std
            noise = torch.max(torch.min(noise, target_noise_clip), -target_noise_clip)
            next_actions = self.actor_target(batch_next_obs) + noise
            low = torch.tensor(self.action_low, device=self.device).unsqueeze(0)
            high = torch.tensor(self.action_high, device=self.device).unsqueeze(0)
            next_actions = torch.max(torch.min(next_actions, high), low)
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
        torch.nn.utils.clip_grad_norm_(self.critic_1.parameters(), self.grad_clip)
        self.critic_1_optimizer.step()

        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_2.parameters(), self.grad_clip)
        self.critic_2_optimizer.step()

        # delayed policy updates
        actor_loss = None
        if step % 2 == 1:
            # update actor network
            act = self.actor(batch_obs)
            actor_loss = -self.critic_1(batch_obs, act).mean()

            # add l2 penalty for abs(action) - max_action
            max_action = torch.tensor(self.action_high, device=self.device).unsqueeze(0)
            l2_penalty = ((act.abs() - max_action) ** 2).mean()
            actor_loss += 1e-3 * l2_penalty

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
            self.actor_optimizer.step()

            # update target networks
            self._update_target_networks(tau=self.tau)

            # print every 500 steps
            if step % 500 == 1:
                print(f"Step {step}: Actor loss = {actor_loss.item():.2f}, Critic 1 loss = {critic_1_loss.item():.2f}, Critic 2 loss = {critic_2_loss.item():.2f}, Noise_STD = {self.get_noise_std(step):.2f}")
        loss_dict = {
            'actor_loss': actor_loss.item() if actor_loss is not None else None,
            'critic_1_loss': critic_1_loss.item(),
            'critic_2_loss': critic_2_loss.item(),
            'noise_std': self.get_noise_std(step),
        }
        return loss_dict

    def decay_epsilon(self, step):
        if step % self.decay_interval == 0 and self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon, self.epsilon_min)
    
    def get_noise_std(self, step):
        frac = min((step/(configs.G_STEPS * 0.9)), 1.0)
        return self.explore_noise_std - (self.explore_noise_std - self.explore_noise_min) * frac
    
    def save_checkpoint(self, path):
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_1_state_dict': self.critic_1.state_dict(),
            'critic_2_state_dict': self.critic_2.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_1_optimizer_state_dict': self.critic_1_optimizer.state_dict(),
            'critic_2_optimizer_state_dict': self.critic_2_optimizer.state_dict(),
        }, path)
    
    def load_checkpoint(self, path):
        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.actor.load_state_dict(checkpoint['actor_state_dict'])
            self.critic_1.load_state_dict(checkpoint['critic_1_state_dict'])
            self.critic_2.load_state_dict(checkpoint['critic_2_state_dict'])
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
            self.critic_1_optimizer.load_state_dict(checkpoint['critic_1_optimizer_state_dict'])
            self.critic_2_optimizer.load_state_dict(checkpoint['critic_2_optimizer_state_dict'])
            print(f"Checkpoint loaded successfully from {path}")
        except Exception as e:
            print(f"Error loading checkpoint from {path}: {e}")
            raise e