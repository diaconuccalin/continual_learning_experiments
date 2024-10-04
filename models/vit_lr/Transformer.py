import torch.nn as nn

from models.vit_lr.TransformerBlock import TransformerBlock


class Transformer(nn.Module):
    def __init__(
        self,
        num_layers,
        dim,
        num_heads,
        tgt_len,
        ff_dim,
        dropout,
        latent_replay_block,
        device,
    ):
        super().__init__()

        # Set latent replay block
        assert (
            -1 <= latent_replay_block < num_layers
        ), "Invalid latent replay block selection."
        self.latent_replay_block = latent_replay_block

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
        is_pattern, x = x

        if is_pattern:
            blocks_to_run = self.blocks
        else:
            assert self.latent_replay_block > -1, "Latent replay block not set."
            blocks_to_run = self.blocks[self.latent_replay_block :]

        for block in blocks_to_run:
            x = block(x)

        return x
