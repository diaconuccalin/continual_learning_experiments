import torch.nn as nn

from models.vit_lr.MultiHeadSelfAttention import MultiHeadSelfAttention
from models.vit_lr.PositionWiseFeedForward import PositionWiseFeedForward


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, tgt_len, ff_dim, dropout, device):
        super().__init__()
        self.tgt_len = tgt_len

        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = MultiHeadSelfAttention(
            dim=dim, n_heads=num_heads, att_dim=dim, device=device
        )

        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.pwff = PositionWiseFeedForward(dim, ff_dim)

    def forward(self, x):
        h = self.norm1(x)
        h = self.attn(h, self.tgt_len)
        h = self.proj(h)
        h = self.drop(h)

        x = x + h

        h = self.norm2(x)
        h = self.pwff(h)
        h = self.drop(h)

        x = x + h

        return x
