import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, capacity = 1000000):
        self.capacity = capacity
        ...
