"""One-layer transformer for modular arithmetic (a + b) mod p."""
import math
import torch
import torch.nn as nn


class ModularTransformer(nn.Module):
    """
    Predicts (a + b) mod p from the sequence [a, b, =].
    Output taken from the readout (=) position after the final layer-norm.
    """

    def __init__(self, p: int, d_model: int = 128, n_heads: int = 4,
                 n_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.p = p
        self.d_model = d_model
        self.embed = nn.Embedding(p + 1, d_model)          # +1 for the '=' token
        self.pos_embed = nn.Parameter(torch.zeros(3, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout, batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=n_layers, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, p, bias=False)
        nn.init.normal_(self.embed.weight, std=0.02)
        nn.init.normal_(self.pos_embed.data, std=0.02)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        B = a.shape[0]
        readout = torch.full((B,), self.p, dtype=torch.long, device=a.device)
        ids = torch.stack([a, b, readout], dim=1)
        h = self.transformer(self.embed(ids) + self.pos_embed)
        return self.head(self.norm(h[:, -1, :]))


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
