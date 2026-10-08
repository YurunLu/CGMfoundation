import torch
import torch.nn as nn
import torch.nn.functional as F
from models.downstream_base_model import CGMBaseModel


def _normalize_state_key(key: str) -> str:
    for prefix in ("module.", "_orig_mod."):
        if key.startswith(prefix):
            key = key[len(prefix) :]
    return key


def _extract_base_state_dict(payload):
    state_dict = payload.get(
        "model_state_dict",
        payload.get("model_state", payload.get("state_dict", payload)),
    )
    if not isinstance(state_dict, dict):
        raise ValueError("Checkpoint does not contain a valid state_dict payload.")

    normalized = {_normalize_state_key(key): value for key, value in state_dict.items()}
    if any(key.startswith("base.") for key in normalized):
        normalized = {
            key[len("base.") :]: value
            for key, value in normalized.items()
            if key.startswith("base.")
        }
    return normalized


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

    def __init__(
        self, pretrained_ckpt_path, num_event_types, event_embed_dim=64, **base_kwargs
    ):
        super().__init__()
        # --- base model (pretrained) ---
        self.base = CGMBaseModel(**base_kwargs)

        # load checkpoint safely (support dict / state_dict)
        # PyTorch 2.6+ defaults weights_only=True, which can fail on older ckpt formats.
        # If you trust the checkpoint source, load with weights_only=False.
        try:
            ckpt = torch.load(
                pretrained_ckpt_path, map_location="cpu", weights_only=False
            )
        except TypeError:
            # older torch versions don't have weights_only
            ckpt = torch.load(pretrained_ckpt_path, map_location="cpu")
        if not isinstance(ckpt, dict):
            raise ValueError(
                f"Unexpected checkpoint payload type: {type(ckpt).__name__}"
            )
        base_state_dict = _extract_base_state_dict(ckpt)
        overlap = set(base_state_dict).intersection(self.base.state_dict())
        if not overlap:
            raise RuntimeError(
                "Pretrained checkpoint does not match CGMBaseModel keys. "
                f"checkpoint={pretrained_ckpt_path}"
            )
        load_result = self.base.load_state_dict(base_state_dict, strict=False)
        print(
            f"[EventFinetuneModel] Loaded pretrained weights from {pretrained_ckpt_path}"
        )
        if load_result.missing_keys or load_result.unexpected_keys:
            print(
                "[EventFinetuneModel] load_state_dict strict=False details: "
                f"missing={list(load_result.missing_keys)} "
                f"unexpected={list(load_result.unexpected_keys)}"
            )

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
            nn.Linear(self.embed_dim, self.embed_dim),
        )

        # learnable attention scoring for past temporal positions (input: per-time embed_dim -> score)
        self.past_attn_score = nn.Linear(self.embed_dim, 1)

        # LayerNorm to stabilize fusion
        self.ln_pre_transformer = nn.LayerNorm(self.embed_dim)
        self.ln_pre_forecast = nn.LayerNorm(self.embed_dim)

        # gating scalars
        self.alpha_past = nn.Parameter(torch.zeros(1))
        self.alpha_future = nn.Parameter(torch.zeros(1))

    def _encode_patches(self, patches, start_idx=0, stride=1):
        base = self.base
        B, N, P = patches.shape

        if getattr(base, "use_cnn", True):
            x = base.cnn(
                patches.view(B * N, 1, P)
            )  # [B*N, embed_dim, P]  (base.cnn produces embed_dim channels)
            x = base.patch_agg(x).view(B, N, self.embed_dim)  # [B, N, embed_dim]
        else:
            x = base.local_encoder(patches)

        if getattr(base, "use_positional", True):
            x = base.pos_embed(x, start_idx=start_idx, stride=stride)

        if getattr(base, "use_temporal", True):
            x = base.temporal_emb(x, start_idx=start_idx, stride=stride)

        return x

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
        past_ev = self.event_emb(past_events)  # linear along E
        # 2) project to model embed dim per time: [B, N, P, embed_dim]
        past_ev_proj = self.past_event_proj(past_ev)

        # --- Compute attention weights along P (time within patch) ---
        # score: [B, N, P, 1] -> squeeze -> [B, N, P]
        scores = self.past_attn_score(past_ev_proj).squeeze(-1)  # [B, N, P]
        # softmax along time dim to get weights
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)  # [B, N, P, 1]
        # weighted sum across P -> [B, N, embed_dim]
        past_summary = (past_ev_proj * weights).sum(dim=2)  # [B, N, embed_dim]

        # --- Patch encoding via base model ---
        x = self._encode_patches(patches, start_idx=start_idx, stride=stride)

        # --- Inject past event context (gated) BEFORE transformer ---
        # gating via sigmoid so influence grows smoothly
        x = x + torch.sigmoid(self.alpha_past) * past_summary
        x = self.ln_pre_transformer(x)

        # --- Transformer encoding (causal) ---
        if getattr(base, "use_transformer", True):
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
