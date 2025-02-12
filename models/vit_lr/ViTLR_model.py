import torch
import torch.nn as nn

from models.vit_lr.PositionalEmbedding1D import PositionalEmbedding1D
from models.vit_lr.Transformer import Transformer


class ViTLR(nn.Module):
    def __init__(
        self,
        device,
        input_size: tuple[int, int] = (128, 128),
        in_channels: int = 3,
        hidden_dimension: int = 768,
        patch_size: tuple[int, int] = (16, 16),
        num_blocks: int = 12,
        num_heads: int = 12,
        ff_dim: int = 3072,
        dropout_rate: float = 0.1,
        num_classes: int = 1000,
        # Currently supports choosing only one of the transformer blocks or -1 for native run
        latent_replay_layer: int = -1,
    ):
        super().__init__()

        # Check if the input image can be split into an exact number of patches of given size
        assert (
            input_size[0] % patch_size[0] == 0
        ), "Incompatible first shape of input and patch sizes."
        assert (
            input_size[1] % patch_size[1] == 0
        ), "Incompatible second shape of input and patch sizes."

        # Generate required layers
        self.seq_len = int(
            (input_size[0] / patch_size[0]) * (input_size[1] / patch_size[1]) + 1
        )

        self.patch_embedding = nn.Conv2d(
            in_channels=in_channels,
            out_channels=hidden_dimension,
            kernel_size=patch_size,
            stride=patch_size,
        )

        self.class_token = nn.Parameter(torch.zeros(1, 1, hidden_dimension))
        self.positional_embedding = PositionalEmbedding1D(
            self.seq_len, hidden_dimension
        )

        self.transformer = Transformer(
            num_blocks=num_blocks,
            dim=hidden_dimension,
            num_heads=num_heads,
            tgt_len=self.seq_len,
            ff_dim=ff_dim,
            dropout=dropout_rate,
            latent_replay_block=latent_replay_layer,
            device=device,
        )

        self.norm = nn.LayerNorm(hidden_dimension, eps=1e-6)
        self.fc = nn.Linear(hidden_dimension, num_classes)

    def forward(self, x, get_activation=False):
        # Check whether the input is a pattern (an original image), or a stored activation
        is_pattern, x = x

        # Store activation if passed
        activation = None
        if get_activation and not is_pattern:
            activation = x.clone().detach()

        if is_pattern:
            b, c, h, w = x.shape

            # b, c, h, w
            x = self.patch_embedding(x)

            # b, dim, nph, npw (number of patches - height and width)
            x = x.flatten(2).transpose(1, 2)

            # b, nph * npw, dim
            x = torch.cat((self.class_token.expand(b, -1, -1), x), dim=1)

            # b, nph * npw + 1, dim
            x = self.positional_embedding(x)

        # b, nph * npw + 1, dim
        assert (
            x.shape[1] == self.seq_len
        ), f"Expected activation with second shape {self.seq_len}, got {x.shape[1]}."
        assert x.shape[2] == self.patch_embedding.out_channels, (
            f"Expected activation with third shape {self.patch_embedding.out_channels}"
            f", got {x.shape[2]}."
        )
        if get_activation and is_pattern:
            x, activation = self.transformer(x=(is_pattern, x), get_activation=True)
        else:
            x = self.transformer(x=(is_pattern, x), get_activation=False)

        # b, nph * npw + 1, dim
        x = torch.tanh(x)

        # b, nph * npw + 1, dim
        x = self.norm(x)[:, 0]

        # b, dim
        x = self.fc(x)

        # b, num_classes
        if not get_activation:
            return x
        else:
            return x, activation

    def set_backbone_requires_grad(
        self, trainable: bool, only_before_lr_layer: bool = False
    ):
        self.patch_embedding.requires_grad_(trainable)
        self.class_token.requires_grad_(trainable)
        self.positional_embedding.requires_grad_(trainable)

        if only_before_lr_layer:
            for i, transformer_block in enumerate(self.transformer.blocks):
                if i == self.transformer.latent_replay_block:
                    break
                transformer_block.requires_grad_(trainable)
        else:
            self.transformer.requires_grad_(trainable)
            self.norm.requires_grad_(trainable)

        for transformer_block in self.transformer.blocks:
            transformer_block.attn.proj_out.requires_grad_(False)
            if transformer_block.attn.proj_out.bias is not None:
                transformer_block.attn.proj_out.bias.requires_grad_(False)

        self.set_layer_norm_trainable()

    def set_layer_norm_trainable(self):
        for transformer_block in self.transformer.blocks:
            transformer_block.norm1.requires_grad_(True)
            transformer_block.norm2.requires_grad_(True)
