import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# 🔥🔥🔥 修改点 1：新增标准的正弦/余弦绝对位置编码类
class PositionalEncoding(nn.Module):
    """
    标准的正弦/余弦绝对位置编码。
    由于参数是固定的数学公式计算得出，彻底免疫 Stride 变化导致的“未学习参数”灾难。
    """
    def __init__(self, embed_dim, max_seq_len=5000):
        super().__init__()
        pe = torch.zeros(max_seq_len, embed_dim)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # 注册为 buffer，这样它会自动跟着模型 to(device)，但不会作为参数被优化器更新
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x, start_idx=0, stride=1):
        device = x.device
        N = x.size(1)
        # 计算每个 Patch 对应的真实物理时间步索引
        abs_idx = torch.arange(N, device=device) * stride + start_idx
        # 防止极端情况越界
        abs_idx = torch.clamp(abs_idx, max=self.pe.size(1) - 1)
        
        # 精准抽取对应物理位置的正弦波编码
        pos_vec = self.pe[0, abs_idx, :] 
        return x + pos_vec.unsqueeze(0)


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
        device = x.device
        N = x.size(1)
        abs_idx = torch.arange(N, device=device) * stride + start_idx
        time_idx = abs_idx % self.steps_per_day
        day_idx = abs_idx // self.steps_per_day

        time_vec = self.time_emb(time_idx)    
        day_vec  = self.day_emb(day_idx)      
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
        w = self.mlp(x.permute(0, 2, 1))   
        w = torch.softmax(w, dim=1)
        return (x * w.permute(0, 2, 1)).sum(dim=-1)  


# --- Base Model ---
class CGMBaseModel(nn.Module):
    def __init__(self,
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
                 use_positional=True): 
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        self.use_cnn = use_cnn                 
        self.use_transformer = use_transformer 
        self.use_temporal = use_temporal     
        self.use_positional = use_positional

        # 保持单通道输入，不做双通道修改
        self.cnn = nn.Sequential(
            nn.Conv1d(1, embed_dim // 2, 6, stride=1, padding=2),
            nn.GELU(),
            nn.Conv1d(embed_dim // 2, embed_dim, 6, stride=1, padding=2),
            nn.GELU(),
        )
        self.patch_agg = PatchAggregator(embed_dim)
        self.local_encoder = nn.Sequential(nn.Linear(self.patch_size, self.embed_dim), nn.GELU())

        # 🔥🔥🔥 修改点 2：将原本的 nn.Parameter 替换为数学固定的正弦位置编码
        # self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, embed_dim))
        # nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.pos_embed = PositionalEncoding(embed_dim, max_seq_len=max_seq_len)

        self.temporal_emb = TemporalEmbedding(embed_dim, steps_per_day=288, max_seq_len=max_seq_len)

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
        self.forecast_head = nn.Linear(embed_dim, forecast_size)

    def forward(self, patches, start_idx=0, stride=1, return_embedding=False):
        B, N, P = patches.shape

        if self.use_cnn:
            x = self.cnn(patches.view(B * N, 1, P))  
            x = self.patch_agg(x).view(B, N, self.embed_dim)
        else:
            x = self.local_encoder(patches)

        # 绝对位置编码（可消融）
        if self.use_positional:
            x = self.pos_embed(x, start_idx=start_idx, stride=stride)

        if self.use_temporal:
            x = self.temporal_emb(x, start_idx=start_idx, stride=stride)

        #if self.use_transformer:
        #   x = self.transformer(x, is_causal=True)
        # Transformer encoder with causal mask
        if self.use_transformer:
            x = self.transformer(x, mask=self.causal_mask[:N, :N])

        if return_embedding:
            return x
        forecast = self.forecast_head(x)
        return forecast