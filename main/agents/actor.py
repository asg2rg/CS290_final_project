# output 2 discrete actions: turn(0~6) and accel(0~5)
import torch
import torch.nn as nn
import utils.configs as configs

class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.tgts_net = nn.Sequential(
            nn.Linear(2, 16),
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

        actor_shape = obs_dim - 2 + 8 - 3 + 8
        # print(f"Actor input shape: {actor_shape}")
        self.network = nn.Sequential(
            nn.Linear(actor_shape, 128),
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
        tgt_in = obs[:, :2]
        tgt_emb = self.tgts_net(tgt_in)
        state_in = obs[:, 2:5]
        state_emb = self.state_net(state_in)
        # remove first 5 dims (target info and lane/speed/yaw), concat with rest of obs
        actor_in = torch.cat([tgt_emb, state_emb, obs[:, 5:]], dim=1)
        # print(f"tgt emb shape: {tgt_emb.shape}, state emb shape: {state_emb.shape}, actor_in shape: {actor_in.shape}")
        return self.network(actor_in) * action_scale if not configs.CLAMP else self.network(actor_in)