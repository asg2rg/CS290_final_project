import numpy as np
import utils.configs as configs
from agents.agent_controller import AgentController

class AgentCar:
    def __init__(self, x, y, heading, speed, id):
        self.state = np.array([x, y, heading], dtype=np.float32)
        self.speed = speed
        self.heading = heading
        self.brains = AgentController(id = id)

    def reset(self, x, y, heading, speed):
        self.state[0] = x
        self.state[1] = y
        self.state[2] = heading # yaw
        self.heading = heading
        self.speed = speed
        self.brains.steps = 0
        self.brains.is_drunk = np.random.rand() < 0.1 if not self.brains.is_drunk else False
    
    def update_state(self, turn_cmd, acc_cmd, dt):
        self.heading += dt * turn_cmd * configs.TURN_UNIT
        if self.heading > np.pi:
            self.heading -= 2 * np.pi
        if self.heading < -np.pi:
            self.heading += 2 * np.pi
        self.speed += acc_cmd
        self.speed = np.clip(self.speed, 5.0, configs.MAX_SPEED)
        # print(f"AgentCar update_state: turn_cmd={turn_cmd}, acc_cmd={acc_cmd}, new_heading={self.heading}, new_speed={self.speed}")

    def move(self, dt):
        if self.brains.just_finished_lane_change:
            self.heading = 0.0
            self.state[2] = 0.0
            self.brains.just_finished_lane_change = False

        self.state[0] += self.speed * dt * np.cos(self.heading)
        self.state[1] += self.speed * dt * np.sin(self.heading)
        self.state[2] = self.heading

    def step(self, dt, obs):
        turn_cmd, acc_cmd = self.brains.make_decision(obs)
        self.update_state(turn_cmd, acc_cmd, dt)
        self.move(dt)