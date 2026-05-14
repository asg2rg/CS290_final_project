# output 2 discrete actions: turn(0~6) and accel(0~5)
import torch
import torch.nn as nn
import utils.configs as configs
from utils.utils import obs_norm, obs_denorm, action_norm, action_denorm

class Critic(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.tgts_net = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU()
        )
        self.state_net = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU()
        )

        critic_shape = obs_dim - 4 + 8 - 3*configs.STACK_SZ + 8*configs.STACK_SZ + action_dim
        # print(f"Critic input shape: {critic_shape}")
        self.network = nn.Sequential(
            nn.Linear(critic_shape, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        # init to low weights
        for m in self.network.modules():
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, -0.003, 0.003)
                nn.init.zeros_(m.bias)

    def forward(self, obs, action):
        if configs.NORM:
            obs = obs_norm(obs)
            action = action_norm(action)
        tgt_in = obs[:, :4]
        tgt_emb = self.tgts_net(tgt_in)
        state_in = obs[:, 4:4 + 3*configs.STACK_SZ]
        state_in = state_in.reshape(-1, 3) # reshape to (B*stack_sz, 3)
        state_emb = self.state_net(state_in)
        state_emb = state_emb.reshape(-1, 8*configs.STACK_SZ) # reshape back to (B, 8*stack_sz)
        # remove first 5 dims (target info and lane/speed/yaw), concat with rest of obs
        critic_in = torch.cat([tgt_emb, state_emb, obs[:, 4 + 3*configs.STACK_SZ:], action], dim=1)
        # print(f"tgt emb shape: {tgt_emb.shape}, state emb shape: {state_emb.shape}, critic_in shape: {critic_in.shape}")
        return self.network(critic_in)