import torch
import torch.nn as nn
import utils.configs as configs

class Critic(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        S = configs.STACK_SZ
        N = configs.NEAREST_AGENTS
        F = configs.OBS_DIM + 1 # exists flag

        self.agent_feat_dim = S * F
        self.agent_embs = nn.Sequential(
            nn.Linear(self.agent_feat_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        self.norm = nn.LayerNorm(64)

        self.car_obs = 4 + S * 3 + (S - 1) * 2
        # critic input = ego_ctx + deepsets_context + action
        self.net = nn.Sequential(
            nn.Linear(self.car_obs + 64 + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        self._S = S
        self._N = N
        self._F = F

    def _parse(self, obs):
        B = obs.shape[0]
        S, N, F = self._S, self._N, self._F
        task    = obs[:, :4]
        ego     = obs[:, 4 : 4 + S*3]
        ag_flat = obs[:, 4 + S*3 : 4 + S*3 + S*N*F]
        actions = obs[:, 4 + S*3 + S*N*F :]
        agents_4d = ag_flat.view(B, S, N, F).permute(0, 2, 1, 3).contiguous()
        ego_ctx = torch.cat([task, ego, actions], dim=-1)
        return ego_ctx, agents_4d

    def forward(self, obs, action):
        B = obs.shape[0]
        S, N, F = self._S, self._N, self._F

        ego_ctx, agents_4d = self._parse(obs)

        exists = agents_4d[:, :, -1, 0:1]
        car_feats = agents_4d.reshape(B, N, self.agent_feat_dim)

        embeddings = self.agent_embs(car_feats)
        embeddings = embeddings * exists
        context = embeddings.sum(dim=1)
        context = self.norm(context)

        x = torch.cat([ego_ctx, context, action], dim=-1)
        return self.net(x)