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
        self.steps = 0
    
    def make_decision(self, obs):
        self.steps += 1
        if self.real_init:
            # Process obs and get action from actor
            action = self.actor(obs)
            turn_cmd, acc_cmd = action[0], action[1]
            return turn_cmd, acc_cmd
        else:
            rand_accel = np.random.uniform(-configs.MAX_ACC, configs.MAX_ACC)
            if self.steps % 30 == 0:
                rand_turn = np.random.uniform(-0.5, 0.5)
            else:
                rand_turn = 0.0
            return rand_turn, rand_accel  # placeholder random action