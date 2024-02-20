import torch
import torchvision
from torch import nn


def get_resnet18():
    model = torchvision.models.resnet18(
        # weights=torchvision.models.ResNet18_Weights.DEFAULT
    )
    return model
