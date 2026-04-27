# output 2 discrete actions: turn(0~6) and accel(0~5)
import torch
import torch.nn as nn
import utils.configs as configs

class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Tanh() if not configs.DISCRETE and configs.CLAMP else nn.Identity()
        )
        # init to low weights
        for m in self.network.modules():
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, -0.003, 0.003)
                nn.init.zeros_(m.bias)

    def forward(self, obs):
        action_scale = torch.tensor([configs.MAX_ANG, configs.MAX_ACC], dtype=obs.dtype, device=obs.device)
        return self.network(obs) * action_scale if not configs.DISCRETE and configs.CLAMP else self.network(obs)