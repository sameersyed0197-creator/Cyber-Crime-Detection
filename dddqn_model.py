import torch
import torch.nn as nn
import random
import numpy as np
from collections import deque

class DuelingDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DuelingDQN, self).__init__()
        
        self.feature = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    
    def forward(self, x):
        x = self.feature(x)
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        
        # Q = V + (A - mean(A))
        return value + advantage - advantage.mean(dim=1, keepdim=True)
    

class PrioritizedReplayBuffer:
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.memory = []
        self.pos = 0
    
    def push(self, transition):
        if len(self.memory) < self.capacity:
            self.memory.append(transition)
        else:
            self.memory[self.pos] = transition
            
        self.pos = (self.pos + 1) % self.capacity
    
    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        return map(np.array, zip(*batch))
    
    def __len__(self):
        return len(self.memory)
