import torch
import numpy as np
import utils.configs as configs
from agents.actor import Actor
from agents.critic import Critic

class AgentController:
    def __init__(self, real_init = False):
        if real_init:
            self.actor = Actor(...)
            self.critic = Critic(...)
        else:
            print("Initialize placeholder Controller")
        self.real_init = real_init
    
    def make_decision(self, obs):
        if self.real_init:
            # Process obs and get action from actor
            action = self.actor(obs)
            turn_cmd, acc_cmd = action[0], action[1]
            return turn_cmd, acc_cmd
        else:
            rand_accel = np.random.uniform(-3.0, 3.0)
            rand_turn = np.random.uniform(-0.5, 0.5)
            return rand_turn, rand_accel  # placeholder random action