import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_absolute_indices(x, start_idx=0, stride=1):
    device = x.device
    num_tokens = x.size(1)
    rel_idx = torch.arange(num_tokens, device=device, dtype=torch.long)
    stride = int(stride)

    if torch.is_tensor(start_idx):
        start_idx = start_idx.to(device=device, dtype=torch.long)
        if start_idx.ndim == 0:
            return rel_idx * stride + start_idx
        if start_idx.ndim == 2:
            if start_idx.shape != (x.size(0), num_tokens):
                raise ValueError(
                    "start_idx tensor with explicit token indices must have shape "
                    f"[{x.size(0)}, {num_tokens}], got {tuple(start_idx.shape)}"
                )
            return start_idx
        if start_idx.ndim != 1 or start_idx.numel() != x.size(0):
            raise ValueError(
                f"start_idx tensor must have shape [{x.size(0)}], got {tuple(start_idx.shape)}"
            )
        return start_idx.view(-1, 1) + rel_idx.view(1, -1) * stride

    return rel_idx * stride + int(start_idx)


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_seq_len=5000):
        super().__init__()
        pe = torch.zeros(max_seq_len, embed_dim)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x, start_idx=0, stride=1):
        abs_idx = _build_absolute_indices(x, start_idx=start_idx, stride=stride)
        abs_idx = torch.clamp(abs_idx, max=self.pe.size(1) - 1)

        if abs_idx.ndim == 1:
            pos_vec = self.pe[0, abs_idx, :]
            return x + pos_vec.unsqueeze(0)

        pos_vec = (
            self.pe[0]
            .index_select(0, abs_idx.reshape(-1))
            .view(abs_idx.size(0), abs_idx.size(1), -1)
        )
        return x + pos_vec


class TemporalEmbedding(nn.Module):
    def __init__(self, embed_dim, steps_per_day=288, max_seq_len=3456, dropout=0.1):
        super().__init__()
        self.steps_per_day = steps_per_day
        self.num_days = math.ceil(max_seq_len / steps_per_day)
        self.time_emb = nn.Embedding(steps_per_day, embed_dim)
        self.day_emb = nn.Embedding(self.num_days, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, start_idx=0, stride=1):
        abs_idx = _build_absolute_indices(x, start_idx=start_idx, stride=stride)
        time_idx = abs_idx % self.steps_per_day
        day_idx = abs_idx // self.steps_per_day

        time_vec = self.time_emb(time_idx)
        day_vec = self.day_emb(day_idx)
        if abs_idx.ndim == 1:
            out = x + time_vec.unsqueeze(0) + day_vec.unsqueeze(0)
        else:
            out = x + time_vec + day_vec
        return self.dropout(out)


class PatchAggregator(nn.Module):
    def __init__(self, embed_dim, hidden_dim=64):
        super().__init__()
        self.query = nn.Parameter(torch.randn(embed_dim))
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        w = self.mlp(x.permute(0, 2, 1))
        w = torch.softmax(w, dim=1)
        return (x * w.permute(0, 2, 1)).sum(dim=-1)


class CGMBaseModel(nn.Module):
    def __init__(
        self,
        patch_size=36,
        forecast_size=36,
        embed_dim=128,
        depth=4,
        num_heads=8,
        hidden_dim=128,
        dropout=0.2,
        max_seq_len=3456,
        use_cnn=True,
        use_transformer=True,
        use_temporal=True,
        use_positional=True,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        self.use_cnn = use_cnn
        self.use_transformer = use_transformer
        self.use_temporal = use_temporal
        self.use_positional = use_positional

        self.cnn = nn.Sequential(
            nn.Conv1d(1, embed_dim // 2, 6, stride=1, padding=2),
            nn.GELU(),
            nn.Conv1d(embed_dim // 2, embed_dim, 6, stride=1, padding=2),
            nn.GELU(),
        )
        self.patch_agg = PatchAggregator(embed_dim)
        self.local_encoder = nn.Sequential(
            nn.Linear(self.patch_size, self.embed_dim),
            nn.GELU(),
        )

        self.pos_embed = PositionalEncoding(embed_dim, max_seq_len=max_seq_len)
        self.temporal_emb = TemporalEmbedding(
            embed_dim,
            steps_per_day=288,
            max_seq_len=max_seq_len,
        )

        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=depth)
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool(),
        )
        self.forecast_head = nn.Linear(embed_dim, forecast_size)

    def forward(self, patches, start_idx=0, stride=1, return_embedding=False):
        bsz, num_tokens, patch_len = patches.shape

        if self.use_cnn:
            x = self.cnn(patches.view(bsz * num_tokens, 1, patch_len))
            x = self.patch_agg(x).view(bsz, num_tokens, self.embed_dim)
        else:
            x = self.local_encoder(patches)

        if self.use_positional:
            x = self.pos_embed(x, start_idx=start_idx, stride=stride)

        if self.use_temporal:
            x = self.temporal_emb(x, start_idx=start_idx, stride=stride)

        if self.use_transformer:
            x = self.transformer(x, mask=self.causal_mask[:num_tokens, :num_tokens])

        if return_embedding:
            return x

        return self.forecast_head(x)
