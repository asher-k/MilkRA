import torch.nn as nn


class CNN(nn.Module):
    """
    Baseline CNN (as an intermediary between MLP & Transformer)
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = nn.Linear(len(x))
        return x
