import numpy as np
import torch
import utils.configs as configs

class ReplayBuffer:
    def __init__(self, capacity = 1000000):
        self.capacity = capacity
        self.idx = 0
        self.sz = 0

        self.state_dim = configs.OBS_DIM
        self.action_dim = configs.ACTION_DIM
        self.obs_dim = 1 + self.state_dim * configs.STACK_SZ + self.action_dim * (configs.STACK_SZ - 1)
        self.obs = np.zeros((capacity, self.obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, self.obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

    def add(self, obs, action, reward, next_obs, done):
        self.obs[self.idx] = obs
        self.next_obs[self.idx] = next_obs
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.dones[self.idx] = done

        self.idx = (self.idx + 1) % self.capacity
        self.sz = min(self.sz + 1, self.capacity)
    
    def sample(self, batch_size, device = 'cpu'):
        idxs = np.random.randint(0, self.sz, size=batch_size)
        batch_obs = torch.FloatTensor(self.obs[idxs]).to(device)
        batch_next_obs = torch.FloatTensor(self.next_obs[idxs]).to(device)
        batch_actions = torch.FloatTensor(self.actions[idxs]).to(device)
        batch_rewards = torch.FloatTensor(self.rewards[idxs]).to(device)
        batch_dones = torch.FloatTensor(self.dones[idxs]).to(device)
        return batch_obs, batch_actions, batch_rewards, batch_next_obs, batch_dones
    
    def __len__(self):
        return self.sz
