# SampleRepresentation/sample_representation_model_mrse.py
import torch
import torch.nn as nn

from models.base_model import CGMBaseModel
from SampleRepresentation.sample_representation_model import load_encoder_ignoring_forecast_head


class ResidualSetEncoder(nn.Module):
    """
    DeepSets-style set encoder for residual patch embeddings.
    Input: residuals [B,N,D] + mask [B,N]
    Output: resid_emb [B,D]
    """
    def __init__(self, embed_dim: int, hidden_dim: int = 256, dropout: float = 0.0):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.rho = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, residuals: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        residuals: [B,N,D]
        mask:      [B,N] boolean or {0,1}
        """
        if mask.dtype != torch.bool:
            mask = mask.bool()
        w = mask.unsqueeze(-1).float()  # [B,N,1]

        x = self.phi(residuals) * w
        denom = w.sum(dim=1).clamp(min=1.0)  # [B,1]
        pooled = x.sum(dim=1) / denom        # [B,D]
        return self.rho(pooled)


class CGMSampleMRSE(nn.Module):
    """
    MRSE model:
      - Pretrained/frozen CGMBaseModel provides patch embeddings H=[B,N,D]
      - Compute global mean (masked) once: mean_emb [B,D]
      - Compute residuals wrt global mean: R=[B,N,D]
      - Residual set encoder produces resid_emb [B,D] from masked residuals
    """
    def __init__(
        self,
        checkpoint_path: str,
        patch_size: int = 36,
        embed_dim: int = 128,
        max_seq_len: int = 3456,
        finetune_encoder: bool = False,
        resid_hidden_dim: int = 256,
        resid_dropout: float = 0.1,
    ):
        super().__init__()

        self.encoder = CGMBaseModel(
            patch_size=patch_size,
            forecast_size=12,
            embed_dim=embed_dim,
            max_seq_len=max_seq_len,
        )

        # loads everything except forecast_head.*
        load_encoder_ignoring_forecast_head(self.encoder, checkpoint_path)

        if not finetune_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        self.residual_encoder = ResidualSetEncoder(
            embed_dim=embed_dim,
            hidden_dim=resid_hidden_dim,
            dropout=resid_dropout,
        )

    @staticmethod
    def masked_mean(h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        h:    [B,N,D]
        mask: [B,N] boolean or {0,1}
        returns: [B,D]
        """
        if mask.dtype != torch.bool:
            mask = mask.bool()
        w = mask.unsqueeze(-1).float()             # [B,N,1]
        denom = w.sum(dim=1).clamp(min=1.0)        # [B,1]
        return (h * w).sum(dim=1) / denom          # [B,D]

    def forward(
        self,
        patches: torch.Tensor,
        patch_mask: torch.Tensor,
        stride: int,
        resid_mask: torch.Tensor = None,
    ):
        """
        patches:     [B,N,P]
        patch_mask:  [B,N] (valid patches)
        resid_mask:  [B,N] (optional; if None uses patch_mask)

        Returns:
          mean_emb  [B,D]
          resid_emb [B,D]
        """
        # Patch embeddings from frozen encoder
        h = self.encoder(patches, stride=stride, return_embedding=True)  # [B,N,D]

        # Global mean computed ONCE per sample from the full patch_mask
        mean_emb = self.masked_mean(h, patch_mask)  # [B,D]

        # Residuals in the fixed coordinate system
        residuals = h - mean_emb.unsqueeze(1)       # [B,N,D]

        if resid_mask is None:
            resid_mask = patch_mask

        resid_emb = self.residual_encoder(residuals, resid_mask)  # [B,D]
        return mean_emb, resid_emb
