import torch
from torch import nn


class MaxPool(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(4)

    def forward(self, x):
        x = self.pool(x)
        return x


def get_default_model():
    return MaxPool()
