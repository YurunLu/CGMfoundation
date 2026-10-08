import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
CODE_ROOT = REPO_ROOT

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from LabelPredictionFinetune import lab_cv_runner as runner
import torch
import torch.nn as nn
import torch.optim as optim


def set_gpu_flags():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def get_layerwise_optimizer(
    model,
    base_lr: float = 1e-5,
    layer_decay: float = 0.7,
    weight_decay: float = 1e-2,
) -> optim.AdamW:
    param_groups = []

    head_params = list(model.pool_attn.parameters()) + list(model.head.parameters())
    param_groups.append({"params": head_params, "lr": base_lr})

    depth = 12
    for name, param in model.base.named_parameters():
        if not param.requires_grad:
            continue
        layer_idx = depth - 1
        for i in range(depth):
            if f".{i}." in name:
                layer_idx = i
                break
        lr_i = base_lr * (layer_decay ** (depth - 1 - layer_idx))
        param_groups.append({"params": [param], "lr": lr_i})

    return optim.AdamW(param_groups, weight_decay=weight_decay)


class FinetuneModelV2(nn.Module):
    def __init__(
        self,
        base_model,
        dropout=0.3,
        model_stride=3,
        num_classes=None,
        regression_dim=None,
        freeze_base=True,
    ):
        super().__init__()
        self.base = base_model
        self.num_classes = num_classes
        self.reg_dim = regression_dim
        self.model_stride = model_stride
        embed_dim = base_model.embed_dim

        if freeze_base:
            for param in self.base.parameters():
                param.requires_grad = False

        self.pool_attn = nn.Linear(embed_dim, 1)
        out_dim = num_classes if num_classes else regression_dim
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, out_dim),
        )

    def forward(self, patches, forecast_mask=None):
        embeddings = self.base(
            patches,
            start_idx=0,
            stride=self.model_stride,
            return_embedding=True,
        )

        scores = self.pool_attn(embeddings).squeeze(-1)
        if forecast_mask is not None:
            scores = scores.masked_fill(~forecast_mask.bool(), float("-inf"))
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        pooled = (embeddings * weights).sum(dim=1)

        out = self.head(pooled)
        if self.num_classes:
            return {"logits": out}
        return {"regression": out}


def configure_runner():
    runner.set_gpu_flags = set_gpu_flags
    runner.FinetuneModel = FinetuneModelV2
    runner.get_layerwise_optimizer = get_layerwise_optimizer
    return runner


def main():
    configure_runner()
    runner.main()


if __name__ == "__main__":
    main()
