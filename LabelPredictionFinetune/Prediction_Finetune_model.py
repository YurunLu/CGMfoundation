# model_finetune.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# expected: from your project import CGMBaseModel
# from models.base_model import CGMBaseModel
# For this snippet assume CGMBaseModel is importable
# from models.base_model import CGMBaseModel

class THeadTransformer(nn.Module):
    """
    Transformer 't-head' that pools token embeddings into a single <cls> vector.
    - Input: embeddings [B, T, D]
    - Masking: src_key_padding_mask with shape [B, T] where True means mask (pad)
    - We prepend a trainable cls token (so transformer sees seq len = T+1)
    - Output: pooled vector [B, D]
    """
    def __init__(self, dim, depth=2, num_heads=8, hidden_dim=256, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=num_heads, dim_feedforward=hidden_dim,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.cls = nn.Parameter(torch.randn(1, 1, dim))  # (1,1,D)

    def forward(self, x, src_key_padding_mask: Optional[torch.Tensor] = None):
        """
        x: [B, T, D]
        src_key_padding_mask: [B, T] bool (True = position is masked/padded)
        returns: pooled vector [B, D]
        """
        B, T, D = x.shape
        cls_tok = self.cls.expand(B, -1, -1)  # [B,1,D]
        x = torch.cat([cls_tok, x], dim=1)  # [B, T+1, D]

        if src_key_padding_mask is not None:
            # transformer expects padding mask shape [B, S], True == masked
            # Need to expand for cls token: it should NOT be masked => prepend False
            prefix = torch.zeros((B, 1), dtype=torch.bool, device=src_key_padding_mask.device)
            padding_mask = torch.cat([prefix, src_key_padding_mask], dim=1)
        else:
            padding_mask = None

        out = self.transformer(x, src_key_padding_mask=padding_mask)  # [B, T+1, D]
        pooled = out[:, 0, :]  # cls position
        return pooled


class FinetuneModel(nn.Module):
    """
    Wrapper that takes CGMBaseModel and applies t-head pooling + heads.
    Supports:
      - classification_head: for binary or multi-class classification
      - regression_head: for single or multi-dim regression
    """
    def __init__(self,
                 base_model: nn.Module,
                 t_head_dim: int,
                 t_head_depth: int = 2,
                 t_head_heads: int = 8,
                 mlp_hidden: int = 256,
                 dropout: float = 0.5,
                 num_classes: Optional[int] = None,
                 regression_dim: Optional[int] = None,
                 freeze_base: bool = False):
        super().__init__()
        self.base = base_model
        self.embedding_dim = self.base.embed_dim  # should match
        assert self.embedding_dim == t_head_dim, "t_head_dim should equal base.embed_dim"

        if freeze_base:
            for p in self.base.parameters():
                p.requires_grad = False

        self.thead = THeadTransformer(dim=self.embedding_dim,
                                     depth=t_head_depth,
                                     num_heads=t_head_heads,
                                     hidden_dim=t_head_dim,
                                     dropout=dropout)

        # small MLP trunk after cls
        self.mlp = nn.Sequential(
            nn.Linear(t_head_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, mlp_hidden // 2),
            nn.GELU(),
        )

        # heads
        self.classification_head = None
        self.regression_head = None
        if num_classes is not None:
            # for binary: num_classes=1 (use BCEWithLogits) or >1 use CrossEntropy over logits
            self.classification_head = nn.Linear(mlp_hidden // 2, num_classes)

        if regression_dim is not None:
            self.regression_head = nn.Linear(mlp_hidden // 2, regression_dim)

    def forward(self, patches, start_idx=0, stride=1, forecast_mask=None):
        """
        patches: [B, N, P]  (as CGMBaseModel expects)
        forecast_mask: optional bool tensor [B, N] where True means valid token
                       (If None, all tokens considered valid)
        returns dict with keys 'logits' (classification if exists) and/or 'regression'
        """
        # get sequence embeddings from base (without forecast head)
        # We expect base.forward(..., return_embedding=True) -> [B, N, D]
        embeddings = self.base(patches, start_idx=start_idx, stride=stride, return_embedding=True)
        # embeddings: [B, N, D]
        # forecast_mask should be bool: True==valid. But Transformer padding mask expects True==masked,
        # so we'll invert when passing src_key_padding_mask.
        if forecast_mask is None:
            # no padding
            src_key_padding_mask = None
        else:
            # forecast_mask: [B, N] bool (True==valid). Make padding mask where True indicates PAD.
            # If forecast_mask is 1 for valid, then padding = ~forecast_mask
            src_key_padding_mask = ~forecast_mask.bool()
        #print(src_key_padding_mask)
        pooled = self.thead(embeddings, src_key_padding_mask=src_key_padding_mask)  # [B, D]
        #pooled = embeddings.mean(dim=1)
        trunk = self.mlp(pooled)  # [B, hidden//2]

        out = {}
        if self.classification_head is not None:
            logits = self.classification_head(trunk)
            out['logits'] = logits

        if self.regression_head is not None:
            preds = self.regression_head(trunk)
            out['regression'] = preds

        return out
