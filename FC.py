""" FC Nueral Network for MNIST"""
import torch
import torch.nn as nn
import torch.nn.functional as F

"""Configuration"""
class Config():
    def __init__(self):
        self.input_size = 28*28
        self.hidden_size = 100
        self.output_size = 10

class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.input_size, config.hidden_size, bias=True)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(config.hidden_size, config.output_size, bias=True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return F.log_softmax(x, -1) 
