# Trajectory/Trajectory_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class TwoHeadTrajectoryModel(nn.Module):
    """
    Two-head velocity model (single-phase training):

      d_pred = normalize(dir_head(z, age, label))         (unit direction)
      s_pred = softplus(speed_head(z, age, label))        (nonnegative speed)
      v_pred = s_pred * d_pred                            (vector velocity)

    Also learns a progress axis w (unit vector) for tau = w·z,
    used by progress ordering and age anchoring losses.

    No loss computed here; use Trajectory_utils.compute_losses_twohead.
    """
    def __init__(
        self,
        embed_dim: int = 128,
        num_labels: int = 3,
        unknown_label_index: int = 3,
        hidden: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_labels = num_labels
        self.unknown_label_index = unknown_label_index

        self.label_emb = nn.Embedding(num_labels + 1, embed_dim)

        in_dim = embed_dim + embed_dim + 1

        # Direction head -> raw direction vector (normalize outside)
        self.dir_head = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim),
        )

        # Speed head -> scalar (Softplus to enforce >=0)
        self.speed_head = nn.Sequential(
            nn.Linear(in_dim, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )
        self.softplus = nn.Softplus()

    def forward(self, z: torch.Tensor, age_value: torch.Tensor, label_id: torch.Tensor):
        """
        z: [B,d]
        age_value: [B,1] (normalized age)
        label_id: [B]
        """
        y_vec = self.label_emb(label_id)
        x = torch.cat([z, y_vec, age_value], dim=1)

        d_raw = self.dir_head(x)
        d_pred = F.normalize(d_raw, dim=1)

        s_raw = self.speed_head(x)                 # [B,1]
        s_pred = self.softplus(s_raw).squeeze(1)   # [B]

        v_pred = d_pred * s_pred.unsqueeze(1)

        return {
            "d_pred": d_pred,   # [B,d] unit
            "s_pred": s_pred,   # [B]
            "v_pred": v_pred,   # [B,d]
        }
