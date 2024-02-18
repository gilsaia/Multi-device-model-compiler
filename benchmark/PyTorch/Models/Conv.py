import torch
from torch import nn


class Conv(nn.Module):
    def __init__(self, input_channel, output_channel) -> None:
        super().__init__()
        self.conv = nn.Conv2d(input_channel, output_channel, 3, 1, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


def get_default_model():
    return Conv(3, 64)
