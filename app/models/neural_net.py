import torch
import torch.nn as nn
import torch.nn.functional as F

class GameAI(nn.Module):
    def __init__(self, input_size=76, output_size=16):
        super(GameAI, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)