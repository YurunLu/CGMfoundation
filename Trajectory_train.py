# Trajectory_train.py
import os
import argparse
import random
import numpy as np

import torch
from torch.utils.data import DataLoader

from Trajectory.Trajectory_model import TwoHeadTrajectoryModel
from Trajectory.Trajectory_utils import (
    load_visits_table,
    build_transitions,
    CGMTrajectoryDataset,
    pad_collate,
    VisitEmbeddingExtractor,
    compute_baselines,
    compute_losses_twohead,
)

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def split_by_user(trans_df, val_ratio=0.1, test_ratio=0.1, seed=42):
    users = trans_df["user_id"].unique().tolist()
    rng = np.random.RandomState(seed)
    rng.shuffle(users)
    n = len(users)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)
    test_users = set(users[:n_test])
    val_users = set(users[n_test:n_test+n_val])
    train_users = set(users[n_test+n_val:])
    return (
        trans_df[trans_df["user_id"].isin(train_users)].copy(),
        trans_df[trans_df["user_id"].isin(val_users)].copy(),
        trans_df[trans_df["user_id"].isin(test_users)].copy(),
    )

@torch.no_grad()
def eval_epoch(model, extractor, loader, device, weights):
    model.eval()
    keys = ["loss","l_dir","l_speed_huber","l_speed_mse","l_vec_mse",
            "cos_mean","speed_pred_mean","speed_obs_mean"]
    logs = {k: 0.0 for k in keys}
    n = 0
    for batch in loader:
        for k in batch:
            batch[k] = batch[k].to(device)

        z = extractor.encode(batch["patches_a"], batch["mask_a"])
        z2 = extractor.encode(batch["patches_b"], batch["mask_b"])

        out = model(z=z, age_value=batch["age_a"], label_id=batch["label_id"])
        lossd = compute_losses_twohead(
            model_out=out,
            z=z, z_next=z2, dt=batch["dt"],
            weights=weights
        )

        bs = z.size(0)
        for k in keys:
            logs[k] += float(lossd[k]) * bs
        n += bs

    for k in keys:
        logs[k] /= max(n, 1)
    return logs

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # Data
    parser.add_argument("--nd_csv", type=str, default="data/SIBIONICS_Dup_ND.csv")
    parser.add_argument("--pred_csv", type=str, default="data/SIBIONICS_Dup_HRT2D.csv")
    parser.add_argument("--t2d_csv", type=str, default="data/SIBIONICS_Dup_T2D.csv")
    parser.add_argument("--info_csv", type=str, default="data/SIBIONIC_Dup_Info.csv")

    # Checkpoints
    parser.add_argument("--base_ckpt", type=str, default="checkpoints/best_forecast_20251012_140643_dim128_patch36_stride6.pt")
    parser.add_argument("--sample_rep_ckpt", type=str, default="checkpoints_sample_rep/best_sample_rep.pt")

    # Patching / Embedding
    parser.add_argument("--patch_size", type=int, default=36)
    parser.add_argument("--stride", type=int, default=6)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--proj_dim", type=int, default=128)
    parser.add_argument("--max_seq_len", type=int, default=3456)

    # Transitions
    parser.add_argument("--min_gap_months", type=float, default=4.0)
    parser.add_argument("--max_pairs_per_user", type=int, default=10)

    # Training
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)

    # Overfitting control
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--min_delta", type=float, default=1e-4)
    parser.add_argument("--min_lr", type=float, default=1e-6)

    # Loss weights (direction is primary)
    parser.add_argument("--w_dir", type=float, default=1.0)
    parser.add_argument("--w_speed", type=float, default=0.5)
    parser.add_argument("--w_vec", type=float, default=0.0)

    # Model params
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)

    parser.add_argument("--out_dir", type=str, default="checkpoints_trajectory")
    args = parser.parse_args()

    set_seed(args.seed)
    device = args.device
    os.makedirs(args.out_dir, exist_ok=True)

    # Visits
    cgm_paths = {"ND": args.nd_csv, "HRT2D": args.pred_csv, "T2D": args.t2d_csv}
    visits = load_visits_table(cgm_paths, args.info_csv, allowed_types=("ND","HRT2D","T2D"))
    print(f"[Visits] n={len(visits)} users={visits['user_id'].nunique()}")

    # Age norm
    ages = visits["age"].dropna()
    age_mean = float(ages.mean()) if len(ages) else 0.0
    age_std = float(ages.std()) if len(ages) else 1.0
    if age_std < 1e-6:
        age_std = 1.0
    print(f"[AgeNorm] mean={age_mean:.3f} std={age_std:.3f}")

    # Transitions
    max_pairs = None if args.max_pairs_per_user <= 0 else args.max_pairs_per_user
    trans = build_transitions(visits, min_gap_months=args.min_gap_months, max_pairs_per_user=max_pairs)
    if len(trans) == 0:
        raise RuntimeError("No transitions after filtering. Check starttime and min_gap_months.")
    print(f"[Transitions] n={len(trans)} dt_months min={trans['dt_months'].min():.2f} med={trans['dt_months'].median():.2f} max={trans['dt_months'].max():.2f}")

    # Split
    train_df, val_df, test_df = split_by_user(trans, args.val_ratio, args.test_ratio, args.seed)
    print(f"[Split] train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    # Datasets
    train_ds = CGMTrajectoryDataset(train_df, patch_size=args.patch_size, stride=args.stride, max_len=args.max_seq_len, age_mean=age_mean, age_std=age_std)
    val_ds   = CGMTrajectoryDataset(val_df,   patch_size=args.patch_size, stride=args.stride, max_len=args.max_seq_len, age_mean=age_mean, age_std=age_std)
    test_ds  = CGMTrajectoryDataset(test_df,  patch_size=args.patch_size, stride=args.stride, max_len=args.max_seq_len, age_mean=age_mean, age_std=age_std)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=4, collate_fn=pad_collate, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=2, collate_fn=pad_collate)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=2, collate_fn=pad_collate)

    # Frozen extractor
    extractor = VisitEmbeddingExtractor(
        base_ckpt_path=args.base_ckpt,
        sample_rep_ckpt_path=args.sample_rep_ckpt,
        patch_size=args.patch_size,
        stride=args.stride,
        embed_dim=args.embed_dim,
        max_seq_len=args.max_seq_len,
        proj_dim=args.proj_dim,
        device=device,
        use_projector=True,
    ).to(device)
    extractor.eval()

    # Baselines
    base = compute_baselines(train_loader, extractor, device=device, max_batches=50, exclude_unknown=True)
    print("[Baseline label counts]", base["label_counts"])
    print("[Baseline: ZERO]")
    print(f"  vec MSE   = {base['zero']['vec_mse']:.4f}")
    print(f"  cos       = {base['zero']['cos']:.4f}")
    print(f"  speed MSE = {base['zero']['speed_mse']:.4f}")
    print("[Baseline: LABEL-MEAN]")
    print(f"  vec MSE   = {base['label_mean']['vec_mse']:.4f}")
    print(f"  cos       = {base['label_mean']['cos']:.4f}")
    print(f"  speed MSE = {base['label_mean']['speed_mse']:.4f}")
    print("[Observed speed] mean={:.4f} std={:.4f}".format(base["obs"]["speed_mean"], base["obs"]["speed_std"]))

    # Model
    model = TwoHeadTrajectoryModel(embed_dim=args.proj_dim, num_labels=3, hidden=args.hidden, dropout=args.dropout).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    weights = {
        "w_dir": args.w_dir,
        "w_speed": args.w_speed,
        "w_vec": args.w_vec,
    }
    print("[Weights]", weights)

    # Scheduler + early stopping
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, mode="min", factor=0.5, patience=5, threshold=args.min_delta, min_lr=args.min_lr
    )

    best_score = float("inf")
    best_epoch = -1
    bad_epochs = 0
    best_path = os.path.join(args.out_dir, "best_twohead.pt")

    # Model selection: direction is primary
    # val_score = l_dir
    def val_score_fn(vlogs):
        return float(vlogs["l_dir"])

    for ep in range(1, args.epochs + 1):
        model.train()
        keys = ["loss","l_dir","l_speed_huber","l_speed_mse","l_vec_mse",
                "cos_mean","speed_pred_mean","speed_obs_mean"]
        tr = {k: 0.0 for k in keys}
        n = 0

        for batch in train_loader:
            for k in batch:
                batch[k] = batch[k].to(device)

            with torch.no_grad():
                z = extractor.encode(batch["patches_a"], batch["mask_a"])
                z2 = extractor.encode(batch["patches_b"], batch["mask_b"])

            out = model(z=z, age_value=batch["age_a"], label_id=batch["label_id"])
            lossd = compute_losses_twohead(
                model_out=out,
                z=z, z_next=z2, dt=batch["dt"],
                weights=weights
            )

            optim.zero_grad(set_to_none=True)
            lossd["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optim.step()

            bs = z.size(0)
            for k in keys:
                tr[k] += float(lossd[k]) * bs
            n += bs

        for k in keys:
            tr[k] /= max(n, 1)

        va = eval_epoch(model, extractor, val_loader, device=device, weights=weights)
        score = val_score_fn(va)
        scheduler.step(score)

        lr_now = optim.param_groups[0]["lr"]

        print(
            f"[Ep {ep:03d}] "
            f"train loss={tr['loss']:.4f} dir={tr['l_dir']:.4f} cos={tr['cos_mean']:.3f} "
            f"speedHuber={tr['l_speed_huber']:.4f} speedMSE={tr['l_speed_mse']:.4f} "
            f"val dir={va['l_dir']:.4f} cos={va['cos_mean']:.3f} "
            f"val speedMSE={va['l_speed_mse']:.4f} score={score:.4f} lr={lr_now:.2e}"
        )

        improved = (best_score - score) > args.min_delta
        if improved:
            best_score = score
            best_epoch = ep
            bad_epochs = 0
            torch.save(
                {"model": model.state_dict(), "args": vars(args), "baseline": base, "best_epoch": best_epoch, "best_score": best_score},
                best_path
            )
        else:
            bad_epochs += 1

        if bad_epochs >= args.patience:
            print(f"[EarlyStop] Best at epoch {best_epoch} score={best_score:.4f}")
            break

    # Test best
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    print(f"[Best] epoch={ckpt.get('best_epoch')} score={ckpt.get('best_score')}")
    te = eval_epoch(model, extractor, test_loader, device=device, weights=weights)
    print(
        f"[Test@Best] dir={te['l_dir']:.4f} cos={te['cos_mean']:.3f} "
        f"speedMSE={te['l_speed_mse']:.4f}"
    )

if __name__ == "__main__":
    train()
