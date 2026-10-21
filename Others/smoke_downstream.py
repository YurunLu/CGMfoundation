"""Check clinical and event-conditioned workflows with synthetic data."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]

def run(workspace):
    workspace = Path(workspace).resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    if (workspace / "LabelPredictionFinetune_CV_main.py").exists():
        raise FileExistsError("Choose a fresh smoke-test directory.")
    records = json.loads((ROOT / "docs/DOWNSTREAM_LAYOUT.json").read_text(encoding="utf-8"))
    for record in records:
        if record["path"].endswith(".py"):
            target = workspace / record["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / record["path"], target)
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    env.pop("CGM_DOWNSTREAM_DATASETS", None)
    # Exercise native Zhao paths in the temporary copy, never real cohort files.
    data_dir = workspace / "data_lab/zhao"
    data_dir.mkdir(parents=True)
    count, length = 48, 48
    rng = np.random.default_rng(42)
    ids = [f"synthetic_{i:03d}" for i in range(count)]
    glucose = 5.5 + np.arange(count)[:, None] % 3 + np.sin(np.arange(length)[None, :] / 8)
    glucose += rng.normal(0, 0.15, (count, length))
    data = pd.DataFrame(glucose, columns=list(map(str, range(length))))
    data.insert(0, "id", ids)
    data.to_csv(data_dir / "Zhao_data.csv", index=False)
    labels = pd.DataFrame({"id": ids, "binary": np.arange(count) % 2,
                           "stage": np.arange(count) % 3, "hba1c": glucose.mean(axis=1)})
    labels.to_csv(data_dir / "Summary_Zhao.csv", index=False)
    labels[["id"]].to_csv(data_dir / "Zhao_label.csv", index=False)
    sys.path.insert(0, str(workspace))
    from models.downstream_base_model import CGMBaseModel
    from GluFormer_ridge_main import TransformerModel
    torch.manual_seed(42)
    hp = dict(patch_size=12, forecast_size=3, embed_dim=16, depth=1, num_heads=2,
              hidden_dim=32, max_seq_len=512, dropout=0.0, use_temporal=False)
    torch.save({"model_state_dict": CGMBaseModel(**hp).state_dict(),
                "model_hparams": {**hp, "forecast_steps": 3, "stride": 6}}, workspace / "random_cgm.pt")
    model = TransformerModel(461, n_embd=16, n_heads=2, n_layers=1,
                             dim_feedforward=32, max_seq_length=512)
    torch.save(model.state_dict(), workspace / "random_gluformer.pt")
    common = dict(dataset="zhao", targets=["cls:binary", "cls:stage", "reg:hba1c"],
                  n_splits=2, min_samples=12, device="cpu", batch_size=8, num_workers=0,
                  split_root=str(workspace / "data_lab/lab_target_cv"), out_root=str(workspace / "outputs"))
    configs = {
        "LabelPredictionFinetune_CV_main.py": {**common, "ckpt_path": str(workspace / "random_cgm.pt"),
            "model_name": "finetune", "epochs": 2, "freeze_base": True,
            "unfreeze_after": 1, "patience": 2, "disable_compile": True},
        "CGMfoundation_ridge_main.py": {**common, "ckpt_path": str(workspace / "random_cgm.pt"),
            "model_name": "cgm_ridge", "save_fold_predictions": True},
        "GluFormer_ridge_main.py": {**common, "ckpt_path": str(workspace / "random_gluformer.pt"),
            "model_name": "glu_ridge", "n_heads": 2, "max_seq_length": 512,
            "glucose_unit": "mmol_l", "save_fold_predictions": True},
        "GMI_IGLU_main.py": {**common, "feature_set": "gmi", "glucose_unit": "mmol_l",
            "save_fold_predictions": True},
    }
    event_data = data.iloc[:12].copy()
    event_data.insert(1, "device_id", [f"record_{i:03d}" for i in range(12)])
    event_data.to_csv(workspace / "event_cgm.csv", index=False)
    pd.DataFrame([{"device_id": device, "TimeIndex": t, "Diet": int(t in (12, 24, 36))}
                  for device in event_data.device_id for t in range(length)]).to_csv(workspace / "events.csv")
    configs["EventFinetune_main.py"] = {
        "run_name": "original_event", "data_csv": str(workspace / "event_cgm.csv"),
        "onehot_csv": str(workspace / "events.csv"), "pretrain_ckpt": str(workspace / "random_cgm.pt"),
        "out_dir": str(workspace / "outputs/event"), "event_cols": ["Diet"], "epochs": 2,
        "freeze_epochs": 1, "batch_size": 4, "num_workers": 0, "device": "cpu",
        "stride": 6, "model_stride": 6, "save_split_datasets": True, "export_predictions": True,
    }
    for script, config in configs.items():
        path = workspace / (Path(script).stem + ".json")
        path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        subprocess.run([sys.executable, str(workspace / script), "--config", str(path)],
                       cwd=workspace, env=env, check=True, timeout=300)
    for model_name in ["finetune", "cgm_ridge", "glu_ridge", "gmi"]:
        results = list((workspace / "outputs" / model_name / "zhao").glob("*/cv_summary.json"))
        if len(results) != 3:
            raise AssertionError(f"Missing results for {model_name}")
    summaries = [p for p in (workspace / "outputs/event").glob("*_summary.json")
                 if not p.name.endswith("_split_summary.json")]
    summary = json.loads(summaries[0].read_text(encoding="utf-8"))
    if not np.isfinite(summary["best_val_loss"]):
        raise AssertionError("Event run did not produce a finite validation loss.")
    if not Path(summary["prediction_export"]["long_path"]).is_file():
        raise AssertionError("Missing event predictions.")
    print("PASS: original clinical, three baseline and event entry points ran on synthetic data.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path)
    args = parser.parse_args()
    if args.work_dir:
        run(args.work_dir)
    else:
        with tempfile.TemporaryDirectory(prefix="cgm_original_") as directory:
            run(Path(directory))
