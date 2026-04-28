import numpy as np

class AgentCar:
    def __init__(self, x, y, heading, speed):
        self.state = np.array([x, y, heading], dtype=np.float32)
        self.speed = speed

    def reset(self, x, y, heading, speed):
        self.state[0] = x
        self.state[1] = y
        self.state[2] = heading
        self.speed = speed

    def step(self, dt):
        heading = self.state[2]
        self.state[0] += self.speed * dt * np.cos(heading)
        self.state[1] += self.speed * dt * np.sin(heading)