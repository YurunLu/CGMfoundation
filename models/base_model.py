import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# --- Sin/Cos Positional Embedding (not learnable) ---
class SinCosPositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=4096):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """x: [B, N, D]"""
        return x + self.pe[:x.size(1)].unsqueeze(0)

class PatchAggregator(nn.Module):
    def __init__(self, embed_dim, hidden_dim=64):
        super().__init__()
        self.query = nn.Parameter(torch.randn(embed_dim))
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        # x: [B, E, P]
        w = self.mlp(x.permute(0, 2, 1))   # [B, P, 1]
        w = torch.softmax(w, dim=1)
        return (x * w.permute(0, 2, 1)).sum(dim=-1)  # [B, E]

# --- Base Model ---
class CGMBaseModel(nn.Module):
    """
    CGM Foundation Model (Forecast version)
    - CNN encoder → positional embedding → causal Transformer → forecast head
    """
    def __init__(self,
                 patch_size=36,
                 forecast_size=36,
                 embed_dim=128,
                 depth=4,
                 num_heads=8,
                 hidden_dim=128,
                 dropout=0.2,
                 max_seq_len=3456):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # CNN patch encoder
        self.cnn = nn.Sequential(
            nn.Conv1d(1, embed_dim // 2, 6, stride=1, padding=2),
            nn.GELU(),
            nn.Conv1d(embed_dim // 2, embed_dim, 6, stride=1, padding=2),
            nn.GELU(),
            #nn.AdaptiveAvgPool1d(1),
        )
        self.patch_agg = PatchAggregator(embed_dim)

        # Position encoding
        self.pos_emb = SinCosPositionalEncoding(embed_dim, max_len=max_seq_len)

        # Causal transformer
        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=depth)
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool()
        )

        # Forecast head
        self.forecast_head = nn.Linear(embed_dim, forecast_size)

    def forward(self, patches):
        """
        patches: [B, N, P]
        return: forecast for each patch [B, N, P]
        """
        B, N, P = patches.shape
        x = self.cnn(patches.view(B * N, 1, P))  # [B*N, E, P]
        #x = F.adaptive_avg_pool1d(x, 1).squeeze(-1).view(B, N, self.embed_dim)
        x = self.patch_agg(x).view(B, N, self.embed_dim)
        #x = self.pos_emb(x)
        x = self.transformer(x, mask=self.causal_mask[:N, :N])
        forecast = self.forecast_head(x)
        return forecast
