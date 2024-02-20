import torch
from torch import nn


class Decoder(nn.Module):
    def __init__(self, d_model, head_num) -> None:
        super().__init__()
        self.encoder = nn.TransformerEncoderLayer(d_model, head_num, batch_first=True)

    def forward(self, x: torch.Tensor):
        shapes = x.shape
        seq = shapes[1]
        mask = nn.Transformer.generate_square_subsequent_mask(seq)
        x = self.encoder(x, mask, is_causal=True)
        return x


def get_qwen1_8B_layer():
    return Decoder(2048, 16)
