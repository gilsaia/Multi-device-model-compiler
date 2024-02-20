import torch
from torch import nn


class SingleLinear(nn.Module):
    def __init__(self, input_channel, output_channel, linear_dim):
        super().__init__()
        self.conv = nn.Conv2d(input_channel, output_channel, 3, 1, 1)
        self.act = nn.ReLU()
        self.linear = nn.Linear(
            output_channel * linear_dim * linear_dim,
            1024,
        )
        self.lineard = nn.Linear(
            1024,
            linear_dim * linear_dim,
        )
        self.act2 = nn.ReLU()
        self.lineart = nn.Linear(
            linear_dim * linear_dim,
            1024,
        )

        self.flat = nn.Linear(1024, output_channel * linear_dim * linear_dim // 4)

    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        x = self.act(x)
        shapes = x.shape
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        y = self.lineard(x)
        y = self.act2(y)
        y = self.lineart(y)
        x = x + y
        x = self.flat(x)
        x = x.view(shapes[0], shapes[1], shapes[2] // 2, shapes[3] // 2)
        return x


class Compat(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.layer1 = SingleLinear(3, 6, 64)
        self.layer2 = SingleLinear(6, 8, 32)
        self.layer3 = SingleLinear(8, 4, 16)
        self.linear = nn.Linear(4 * 8 * 8, 1000)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        return x


def get_default_model():
    return Compat()
