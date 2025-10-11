# models/event_model.py
import torch
import torch.nn as nn
from .base_model import CGMBaseModel, MLP

class CGMEventModel(CGMBaseModel):
    """
    Event-aware model based on CGMBaseModel.
    We apply lightweight FiLM (learn gamma/beta from event features)
    to the patch embeddings BEFORE positional encoding / transformer.
    """

    def __init__(self, event_dim, event_emb_dim=64, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.event_mlp = MLP(event_dim, event_emb_dim, event_emb_dim)
        self.gamma = nn.Linear(event_emb_dim, self.embed_dim)
        self.beta = nn.Linear(event_emb_dim, self.embed_dim)

    def forward(self, patches, event_feats=None, return_tokens=False):
        """
        patches: [B,N,P]
        event_feats: [B,N,D_e] or None
        """
        B, N, P = patches.shape
        x = self.encode_patches(patches)  # [B,N,E]

        # FiLM modulation if events provided
        if event_feats is not None:
            ev = self.event_mlp(event_feats)   # [B,N,Ev]
            gamma = self.gamma(ev)
            beta = self.beta(ev)
            x = gamma * x + beta

        # positional / transformer / heads as base
        x = self.pos_emb(x)
        causal_mask = self.causal_mask[:N, :N].to(x.device)
        x = self.transformer(x, mask=causal_mask)
        recon = self.recon_head(x)
        forecast = self.forecast_head(x)
        if return_tokens:
            return recon, forecast, x
        else:
            return recon, forecast
