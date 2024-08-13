import torch.nn as nn

from models.vit_lr.TransformerBlock import TransformerBlock


class Transformer(nn.Module):
    def __init__(self, num_layers, dim, num_heads, tgt_len, ff_dim, dropout, device):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    tgt_len=tgt_len,
                    ff_dim=ff_dim,
                    dropout=dropout,
                    device=device,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)

        return x
