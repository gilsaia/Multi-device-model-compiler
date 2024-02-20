import torch
from torch import nn


class Relu(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.act(x)
        return x


def get_default_model():
    return Relu()
