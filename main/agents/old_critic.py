# output 2 discrete actions: turn(0~6) and accel(0~5)
import torch
import torch.nn as nn

class Critic(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, obs, action):
        st = torch.cat([obs, action], dim=-1)
        return self.network(st)