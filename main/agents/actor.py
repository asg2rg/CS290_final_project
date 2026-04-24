# output 2 discrete actions: turn(0~6) and accel(0~5)
import torch
import torch.nn as nn

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        ...

    def forward(self, state):
        ...