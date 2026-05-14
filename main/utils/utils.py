import torch
import numpy as np
import utils.configs as configs

# obs order: 
# target lane, target speed, agents front, agents behind, (car lane, car speed, car yaw)*4, (exists, lane, speed, yaw, rel_dist, heading_error)*4, (angular, linear)*3
# range:
# 0~3, 0~100 0~5, 0~5, (-1~3, 0~100, -np.pi~np.pi)*4, (0~1, -1~3, 0~100, -np.pi~np.pi, -500~500, -np.pi~np.pi)*4, (-3~3, -10~10)*3

def obs_norm(obs_t):
    if type(obs_t) is not torch.Tensor:
        obs_t = torch.tensor(obs_t, dtype=torch.float32)
    else:
        obs_t = obs_t.clone()  # Clone to avoid in-place modifications affecting the computational graph
    # normalize obs to range [-1, 1]
    obs_t[:, :2] = obs_t[:, :2] / torch.tensor([configs.MAX_SPEED, configs.ROAD_CNT - 1], device=obs_t.device) * 2 - 1
    obs_t[:, 2:4] = obs_t[:, 2:4] / torch.tensor([configs.MAX_AGENTS, configs.MAX_AGENTS], device=obs_t.device) * 2 - 1
    car_state_end = 4 + 3 * configs.STACK_SZ
    obs_t[:, 4:car_state_end] = obs_t[:, 4:car_state_end] / torch.tensor([configs.ROAD_CNT - 1, configs.MAX_SPEED, np.pi] * configs.STACK_SZ, device=obs_t.device) * 2 - 1
    agent_state_end = car_state_end + 6 * configs.NEAREST_AGENTS * configs.STACK_SZ
    obs_t[:, car_state_end:agent_state_end] = obs_t[:, car_state_end:agent_state_end] / torch.tensor([1, configs.ROAD_CNT - 1, configs.MAX_SPEED, np.pi, 500, np.pi] * (configs.NEAREST_AGENTS * configs.STACK_SZ), device=obs_t.device) * 2 - 1
    action_state_end = agent_state_end + 2 * (configs.STACK_SZ - 1)
    obs_t[:, agent_state_end:] = obs_t[:, agent_state_end:] / torch.tensor([configs.MAX_ANG, configs.MAX_ACC] * (configs.STACK_SZ - 1), device=obs_t.device) * 2 - 1
    return obs_t

def obs_denorm(obs_t):
    if type(obs_t) is not torch.Tensor:
        obs_t = torch.tensor(obs_t, dtype=torch.float32)
    else:
        obs_t = obs_t.clone()  # Clone to avoid in-place modifications affecting the computational graph
    obs_t[:, :2] = (obs_t[:, :2] + 1) / 2 * torch.tensor([configs.MAX_SPEED, configs.ROAD_CNT - 1], device=obs_t.device)
    obs_t[:, 2:4] = (obs_t[:, 2:4] + 1) / 2 * torch.tensor([configs.MAX_AGENTS, configs.MAX_AGENTS], device=obs_t.device)
    car_state_end = 4 + 3 * configs.STACK_SZ
    obs_t[:, 4:car_state_end] = (obs_t[:, 4:car_state_end] + 1) / 2 * torch.tensor([configs.ROAD_CNT - 1, configs.MAX_SPEED, np.pi] * configs.STACK_SZ, device=obs_t.device)
    agent_state_end = car_state_end + 6 * configs.NEAREST_AGENTS * configs.STACK_SZ
    obs_t[:, car_state_end:agent_state_end] = (obs_t[:, car_state_end:agent_state_end] + 1) / 2 * torch.tensor([1, configs.ROAD_CNT - 1, configs.MAX_SPEED, np.pi, 500, np.pi] * (configs.NEAREST_AGENTS * configs.STACK_SZ), device=obs_t.device)
    action_state_end = agent_state_end + 2 * (configs.STACK_SZ - 1)
    obs_t[:, agent_state_end:] = (obs_t[:, agent_state_end:] + 1) / 2 * torch.tensor([configs.MAX_ANG, configs.MAX_ACC] * (configs.STACK_SZ - 1), device=obs_t.device)
    return obs_t

def action_norm(action_t):
    if type(action_t) is not torch.Tensor:
        action_t = torch.tensor(action_t, dtype=torch.float32)
    else:
        action_t = action_t.clone()  # Clone to avoid in-place modifications affecting the computational graph
    return action_t / torch.tensor([configs.MAX_ANG, configs.MAX_ACC], device=action_t.device) * 2 - 1

def action_denorm(action_t):
    if type(action_t) is not torch.Tensor:
        action_t = torch.tensor(action_t, dtype=torch.float32)
    else:
        action_t = action_t.clone()  # Clone to avoid in-place modifications affecting the computational graph
    return (action_t + 1) / 2 * torch.tensor([configs.MAX_ANG, configs.MAX_ACC], device=action_t.device)