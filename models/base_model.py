import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# --- Temporal Embedding (Time-of-day + Day index) ---
class TemporalEmbedding(nn.Module):
    """
    Adds learned embeddings for:
      - time-of-day (288 points per day for 5-min CGM)
      - day index (e.g., up to 12 days)
    Both are added to the patch embeddings.
    """
    def __init__(self, embed_dim, steps_per_day=288, max_seq_len=3456, dropout=0.1):
        super().__init__()
        self.steps_per_day = steps_per_day
        self.num_days = math.ceil(max_seq_len / steps_per_day)
        self.time_emb = nn.Embedding(steps_per_day, embed_dim)
        self.day_emb = nn.Embedding(self.num_days, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, start_idx=0, stride=1):
        """
        x: [B, N, D]  (patch embeddings)
        start_idx: index (in time steps) of first patch, typically 0
        stride: hop length between patches (in 5-min steps)
        """
        device = x.device
        N = x.size(1)

        # absolute index of each patch center (approximation)
        abs_idx = torch.arange(N, device=device) * stride + start_idx

        # time within the day (0..287)
        time_idx = abs_idx % self.steps_per_day
        # which day (0..num_days-1)
        day_idx = abs_idx // self.steps_per_day

        time_vec = self.time_emb(time_idx)    # [N, D]
        day_vec  = self.day_emb(day_idx)      # [N, D]
        out = x + time_vec.unsqueeze(0) + day_vec.unsqueeze(0)
        return self.dropout(out)


# --- Patch Aggregator ---
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
    - CNN encoder → temporal embedding → causal Transformer → forecast head
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
        )
        self.patch_agg = PatchAggregator(embed_dim)

        # --- Replace sinusoidal with temporal embedding ---
        self.temporal_emb = TemporalEmbedding(embed_dim, steps_per_day=288, max_seq_len=max_seq_len)

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

    def forward(self, patches, start_idx=0, stride=1, return_embedding=False):
        """
        patches: [B, N, P]
        return: forecast for each patch [B, N, forecast_size]
        """
        B, N, P = patches.shape
        x = self.cnn(patches.view(B * N, 1, P))  # [B*N, E, P]
        x = self.patch_agg(x).view(B, N, self.embed_dim)

        # Add day/time embedding
        x = self.temporal_emb(x, start_idx=start_idx, stride=stride)

        # Transformer encoder with causal mask
        x = self.transformer(x, mask=self.causal_mask[:N, :N])
        if return_embedding:
            return x
        forecast = self.forecast_head(x)
        return forecast
