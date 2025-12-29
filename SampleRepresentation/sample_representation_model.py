# sample_representation_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base_model import CGMBaseModel


def load_encoder_ignoring_forecast_head(encoder: nn.Module, checkpoint_path: str):
    state = torch.load(checkpoint_path, map_location="cpu")

    # If your checkpoint was saved as {"model": state_dict, ...}, handle that too.
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict) and "model" in state:
        state = state["model"]

    # Drop forecast head params to avoid shape mismatch
    filtered = {k: v for k, v in state.items() if not k.startswith("forecast_head.")}

    missing, unexpected = encoder.load_state_dict(filtered, strict=False)
    # missing should include forecast_head.* (expected)
    return missing, unexpected


class AttnPoolAggregator(nn.Module):
    def __init__(self, embed_dim, num_heads=8):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, mask):
        B, _, E = x.shape
        q = self.query.expand(B, 1, E)
        key_padding_mask = ~mask
        out, _ = self.attn(q, x, x, key_padding_mask=key_padding_mask, need_weights=False)
        return self.norm(out[:, 0])


class CGMSampleRepresentationModel(nn.Module):
    def __init__(
        self,
        checkpoint_path,
        patch_size=36,
        embed_dim=128,
        max_seq_len=3456,
        proj_dim=128,
        finetune_encoder=False,
    ):
        super().__init__()

        # IMPORTANT: forecast_size can be ANYTHING here because we ignore forecast_head in loading.
        self.encoder = CGMBaseModel(
            patch_size=patch_size,
            forecast_size=12,      # set to 12 to match typical pretrain, but not required with filtering
            embed_dim=embed_dim,
            max_seq_len=max_seq_len,
        )

        missing, unexpected = load_encoder_ignoring_forecast_head(self.encoder, checkpoint_path)
        print("[Load ckpt] missing keys (ok):", [k for k in missing if k.startswith("forecast_head.")])
        if len(unexpected) > 0:
            print("[Load ckpt] unexpected keys:", unexpected)

        if not finetune_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        self.aggregator = AttnPoolAggregator(embed_dim)

        self.projector = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, proj_dim),
        )

    def forward(self, patches, patch_mask, view1_mask, view2_mask, stride):
        x = self.encoder(patches, stride=stride, return_embedding=True)  # [B,N,E]

        mask1 = patch_mask & view1_mask
        mask2 = patch_mask & view2_mask

        z1 = self.aggregator(x, mask1)
        z2 = self.aggregator(x, mask2)

        p1 = F.normalize(self.projector(z1), dim=-1)
        p2 = F.normalize(self.projector(z2), dim=-1)
        return p1, p2
