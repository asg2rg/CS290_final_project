import numpy as np
import utils.configs as configs
from agents.agent_controller import AgentController

class AgentCar:
    def __init__(self, x, y, heading, speed):
        self.state = np.array([x, y, heading], dtype=np.float32)
        self.speed = speed
        self.heading = heading
        self.brains = AgentController()

    def reset(self, x, y, heading, speed):
        self.state[0] = x
        self.state[1] = y
        self.state[2] = heading # yaw
        self.heading = heading
        self.speed = speed
        self.brains.steps = 0
    
    def update_state(self, turn_cmd, acc_cmd):
        self.heading += turn_cmd
        self.speed += acc_cmd
        self.speed = np.clip(self.speed, configs.MIN_SPEED, configs.MAX_SPEED)

    def step(self, dt, obs):
        turn_cmd, acc_cmd = self.brains.make_decision(obs)
        self.update_state(turn_cmd, acc_cmd)
        self.state[0] += self.speed * dt * np.cos(self.heading)
        self.state[1] += self.speed * dt * np.sin(self.heading)
        self.state[2] = self.heading