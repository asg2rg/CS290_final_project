# output 2 discrete actions: turn(0~6) and accel(0~5)
import torch
import torch.nn as nn
import utils.configs as configs
from utils.utils import obs_norm, obs_denorm

class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.tgts_net = nn.Sequential( # tgt_speed, tgt_lane
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU()
        )

        self.agent_pos_net = nn.Sequential( # agent_front, agent_back
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU()
        )

        self.state_net = nn.Sequential( # lane, speed, yaw, 4 timesteps
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU()
        )

        self.agents_net = nn.Sequential( # 6 dims per agent * 4 agents, 4 timesteps
            nn.Linear(6 * configs.NEAREST_AGENTS, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 8),
            nn.ReLU()
        )

        self.car_act_net = nn.Sequential( # linear, angular, 3 timesteps
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU()
        )

        actor_shape = 8 + 8 + 8*configs.STACK_SZ + 8*configs.STACK_SZ + 8*(configs.STACK_SZ -1)

        self.network = nn.Sequential(
            nn.Linear(actor_shape, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Tanh() if configs.CLAMP else nn.Identity()
        )
        # init to low weights
        for m in self.network.modules():
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, -0.003, 0.003)
                nn.init.zeros_(m.bias)

    def forward(self, obs):
        if configs.NORM:
            obs = obs_norm(obs)
        action_scale = torch.tensor([configs.MAX_ANG, configs.MAX_ACC], dtype=obs.dtype, device=obs.device)
        tgt_in = obs[:, :2]
        # print(f"tgt_in: {tgt_in[0]}")
        tgt_emb = self.tgts_net(tgt_in)
        pos_in = obs[:, 2:4]
        # print(f"pos_in: {pos_in[0]}")
        pos_emb = self.agent_pos_net(pos_in)
        state_in = obs[:, 4:16] # 4~6, 7~9, 10~12, 13~15 are lane/speed/yaw for 4 timesteps
        state_in = state_in.reshape(-1, configs.STACK_SZ, 3)
        # print(f"state_in: {state_in[0]}")
        state_emb = self.state_net(state_in.reshape(-1, 3)) # shape (batch*stack_sz, 8)
        # reshape back to (batch, stack_sz*8)
        state_emb = state_emb.reshape(-1, configs.STACK_SZ * 8)
        agents_in = obs[:, 16:16+6*configs.NEAREST_AGENTS * configs.STACK_SZ] # 6 dims per agent * 4 agents, 4 timesteps
        agents_in = agents_in.reshape(-1, configs.STACK_SZ, 6 * configs.NEAREST_AGENTS)
        # print(f"agents_in: {agents_in[0]}") # print first batch
        agents_emb = self.agents_net(agents_in.reshape(-1, 6 * configs.NEAREST_AGENTS)) # shape (batch*stack_sz, 8)
        agents_emb = agents_emb.reshape(-1, configs.STACK_SZ * 8)
        car_act_in = obs[:, 16+6*configs.NEAREST_AGENTS * configs.STACK_SZ:] # linear, angular, 3 timesteps
        car_act_in = car_act_in.reshape(-1, configs.STACK_SZ - 1, 2)
        # print(f"car_act_in: {car_act_in[0]}") # print first batch
        car_act_emb = self.car_act_net(car_act_in.reshape(-1, 2))
        car_act_emb = car_act_emb.reshape(-1, (configs.STACK_SZ - 1) * 8)

        actor_in = torch.cat([tgt_emb, pos_emb, state_emb, agents_emb, car_act_emb], dim=-1)
        # print(f"tgt emb shape: {tgt_emb.shape}, state emb shape: {state_emb.shape}, actor_in shape: {actor_in.shape}")
        return self.network(actor_in) * action_scale if configs.CLAMP else self.network(actor_in)