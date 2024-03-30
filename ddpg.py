import gym
import random 
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Hyperparameters
lr_mu = 0.0005
lr_q = 0.001
gamma = 0.99
tau = 0.005 # for target network soft update
batch_size = 32
buffer_limit = 50000

class ReplayBuffer:
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)
        
    def put(self, transition):
        self.buffer.append(transition)
        
    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        