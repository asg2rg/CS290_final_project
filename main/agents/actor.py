import torch
import torch.nn as nn
import utils.configs as configs

class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        S = configs.STACK_SZ
        F = configs.OBS_DIM + 1 # exists flag
        N = configs.NEAREST_AGENTS
        
        # encodes one agent's history
        self.agents_obs_dim = S * F
        self.agent_emb = nn.Sequential(
            nn.Linear(self.agents_obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        
        car_dim = 4 + S * 3 + (S - 1) * 2
        self.net = nn.Sequential(
            nn.Linear(car_dim + 64, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh() if configs.CLAMP else nn.Identity()
        )
        
        self.max_action = torch.tensor([configs.MAX_ANG, configs.MAX_ACC], dtype=torch.float32)
        self._S = S
        self._N = N

    def _parse(self, obs):
        """Split flat obs into (ego_context, agents_4d) tensors."""
        B = obs.shape[0]
        S, N = self._S, self._N
        
        task = obs[:, :4]                                         # (B, 4)
        ego = obs[:, 4 : 4 + S*3]                               # (B, S*3)
        ag_flat = obs[:, 4 + S*3 : 4 + S*3 + S*N*6]                # (B, S*N*6)
        actions = obs[:, 4 + S*3 + S*N*6 :]                         # (B, (S-1)*2)

        # Reshape agents: (B, S, N, 6) → (B, N, S, 6)
        agents_4d = ag_flat.view(B, S, N, 6).permute(0, 2, 1, 3).contiguous()

        ego_ctx = torch.cat([task, ego, actions], dim=-1)            # (B, car_dim)
        return ego_ctx, agents_4d

    def forward(self, obs):
        B = obs.shape[0]
        S, N = self._S, self._N

        ego_ctx, agents_4d = self._parse(obs)                        # (B, car_dim), (B, N, S, 6)

        # exists: use latest timestep's flag for the aggregate mask
        exists = agents_4d[:, :, -1, 0:1]                           # (B, N, 1)

        # Per-car temporal feature
        car_feats = agents_4d.reshape(B, N, self.agents_obs_dim)                  # (B, N, S*6)
        # print(f"car_feats shape: {car_feats.shape}")
        # print(car_feats[0])
        # print(exists[0])

        # DeepSets
        embeddings = self.agent_emb(car_feats)                             # (B, N, 64)
        embeddings = embeddings * exists                             # zero out non-existent
        context = embeddings.sum(dim=1)                              # (B, 64)

        x = torch.cat([ego_ctx, context], dim=-1)
        
        action = self.net(x)
        # scale from tanh [-1,1] to action range, or raw output
        max_action = self.max_action.to(obs.device)
        return action * max_action if configs.CLAMP else action