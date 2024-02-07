import torch
from torch import nn


class Linear(nn.Module):
    def __init__(self, input_feature, output_feature) -> None:
        super().__init__()
        self.linear = nn.Linear(input_feature, output_feature)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.linear(x)
        x = self.act(x)
        return x


def get_default_model():
    return Linear(640, 640)
