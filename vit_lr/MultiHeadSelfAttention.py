import torch
import torch.nn as nn

from vit_lr.SoftmaxFastExp import SoftmaxFastExp
from vit_lr.utils import q_rsqrt


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, n_heads, att_dim):
        super().__init__()

        self.n_heads = n_heads
        self.att_dim = att_dim
        self.head_dim = att_dim // n_heads

        self.proj_in = nn.Linear(dim, 3 * att_dim, bias=False)
        self.scaling = q_rsqrt(self.head_dim)
        self.softmax = SoftmaxFastExp
        self.proj_out = nn.Linear(att_dim, dim, bias=False)

    def forward(self, x, tgt_len):
        # OP 1
        qkv = self.proj_in(x)

        # OP 2
        q = qkv[..., : int(qkv.shape[-1] / 3)]
        k = qkv[..., int(qkv.shape[-1] / 3) : 2 * int(qkv.shape[-1] / 3)]
        v = qkv[..., 2 * int(qkv.shape[-1] / 3) :]

        q = q.contiguous().view(tgt_len, self.n_heads, self.head_dim).transpose(0, 1)
        k = k.contiguous().view(tgt_len, self.n_heads, self.head_dim).transpose(0, 1)
        v = v.contiguous().view(tgt_len, self.n_heads, self.head_dim).transpose(0, 1)

        # OP 3
        scores = torch.bmm(q, k.transpose(1, 2))

        assert list(scores.size()) == [self.n_heads, tgt_len, tgt_len]

        # OP 4
        scores = scores * self.scaling

        # OP 5
        # scores = self.softmax(scores)
        scores = SoftmaxFastExp.apply(scores)

        # OP 6
        scores = torch.bmm(scores, v)
        assert list(scores.size()) == [self.n_heads, tgt_len, self.head_dim]

        scores = scores.transpose(0, 1).contiguous().view(tgt_len, self.att_dim)

        # OP 7
        h = self.proj_out(scores)
        return h
