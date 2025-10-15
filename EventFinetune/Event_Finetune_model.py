import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base_model import CGMBaseModel

class EventFinetuneModel(nn.Module):
    """
    Fine-tune version of CGMBaseModel that integrates event effects.
    - Uses pretrained weights (loaded with strict=False)
    - Past event context: learnable temporal attention over P (per-patch past window),
      projected into embed_dim and added to patch embeddings BEFORE transformer.
    - Future event context: encoded and projected to embed_dim, added to transformer
      outputs (latent x) BEFORE forecast_head (so forecast_head can use event-informed latent).
    - Uses sigmoid gating for both past and future effects so influence starts near 0.
    """
    def __init__(self, pretrained_ckpt_path, num_event_types, event_embed_dim=64, **base_kwargs):
        super().__init__()
        # --- base model (pretrained) ---
        self.base = CGMBaseModel(**base_kwargs)

        # load checkpoint safely (support dict / state_dict)
        ckpt = torch.load(pretrained_ckpt_path, map_location='cpu')
        # ckpt may be a dict with 'model_state' or 'state_dict'
        state_dict = ckpt.get('model_state', ckpt.get('state_dict', ckpt))
        # allow missing/new keys
        self.base.load_state_dict(state_dict, strict=False)
        print(f"[EventFinetuneModel] Loaded pretrained weights from {pretrained_ckpt_path}")

        # dims
        self.embed_dim = self.base.embed_dim
        self.forecast_size = self.base.forecast_head.out_features
        self.num_event_types = num_event_types
        self.event_embed_dim = event_embed_dim

        # --- Event encoders ---
        # map raw event vector (E) -> small event embedding (Ev)
        self.event_emb = nn.Linear(num_event_types, event_embed_dim)

        # project per-time event embedding to model embed dim (for past and future separately)
        self.past_event_proj = nn.Linear(event_embed_dim, self.embed_dim)
        self.future_event_proj = nn.Linear(event_embed_dim, self.embed_dim)

        # small stack for future -> optionally produce richer signal (kept simple)
        self.future_mlp = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, self.embed_dim)
        )

        # learnable attention scoring for past temporal positions (input: per-time embed_dim -> score)
        self.past_attn_score = nn.Linear(self.embed_dim, 1)

        # LayerNorm to stabilize fusion
        self.ln_pre_transformer = nn.LayerNorm(self.embed_dim)
        self.ln_pre_forecast = nn.LayerNorm(self.embed_dim)

        # gating scalars (start near 0 influence). Use sigmoid when applying.
        self.alpha_past = nn.Parameter(torch.zeros(1))
        self.alpha_future = nn.Parameter(torch.zeros(1))

    def forward(self, patches, events, start_idx=0, stride=1):
        """
        patches: [B, N, P]
        events:  [B, N, P+F, E]
        returns:
            forecast: [B, N, F]
        """
        B, N, P = patches.shape
        PF = events.shape[2]
        E = events.shape[3]
        F = PF - P  # forecast horizon (should match forecast_size)

        base = self.base

        # --- Split past and future event windows ---
        # past_events: [B, N, P, E]
        # future_events: [B, N, F, E]
        past_events = events[:, :, :P, :]
        future_events = events[:, :, P:, :]

        # --- Encode past events per timepoint and project to embed_dim ---
        # 1) raw event embedding per time: [B, N, P, event_embed_dim]
        past_ev = self.event_emb(past_events)              # linear along E
        # 2) project to model embed dim per time: [B, N, P, embed_dim]
        past_ev_proj = self.past_event_proj(past_ev)

        # --- Compute attention weights along P (time within patch) ---
        # score: [B, N, P, 1] -> squeeze -> [B, N, P]
        scores = self.past_attn_score(past_ev_proj).squeeze(-1)  # [B, N, P]
        # softmax along time dim to get weights
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)    # [B, N, P, 1]
        # weighted sum across P -> [B, N, embed_dim]
        past_summary = (past_ev_proj * weights).sum(dim=2)       # [B, N, embed_dim]

        # --- Patch encoding via base CNN + aggregation ---
        x = base.cnn(patches.view(B * N, 1, P))    # [B*N, embed_dim, P]  (base.cnn produces embed_dim channels)
        x = base.patch_agg(x).view(B, N, self.embed_dim)  # [B, N, embed_dim]

        # --- Add temporal embedding ---
        x = base.temporal_emb(x, start_idx=start_idx, stride=stride)

        # --- Inject past event context (gated) BEFORE transformer ---
        # gating via sigmoid so influence grows smoothly
        x = x + torch.sigmoid(self.alpha_past) * past_summary
        x = self.ln_pre_transformer(x)

        # --- Transformer encoding (causal) ---
        x = base.transformer(x, mask=base.causal_mask[:N, :N])

        # --- Encode future events (per future time), aggregate & project to embed_dim ---
        # future_events: [B, N, F, E] -> embed -> [B, N, F, event_embed_dim]
        future_ev = self.event_emb(future_events)
        # project to embed_dim per future time: [B, N, F, embed_dim]
        future_ev_proj = self.future_event_proj(future_ev)
        # aggregate across future horizon: use simple mean (can be replaced by attention)
        future_summary = future_ev_proj.mean(dim=2)  # [B, N, embed_dim]
        future_summary = self.future_mlp(future_summary)  # [B, N, embed_dim]

        # --- Inject future event context into latent x BEFORE forecast head ---
        x = x + torch.sigmoid(self.alpha_future) * future_summary
        x = self.ln_pre_forecast(x)

        # --- Forecast head (unchanged) ---
        forecast = base.forecast_head(x)  # [B, N, F]

        return forecast
