import argparse
import json
import os
import signal
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path

DEFAULT_BASE_CONFIG = {
    "run_name": "event_finetune",
    "data_csv": "",
    "onehot_csv": "",
    "pretrain_ckpt": "/workspace/last/best_forecast_20260428_003310_dim256_patch144_stride3.pt",
    "out_dir": "data_event/event_finetune_outputs",
    "epochs": 500,
    "batch_size": 32,
    "patch_size": 144,
    "stride": 3,
    "forecast_steps": 12,
    "embed_dim": 256,
    "depth": 12,
    "num_heads": 8,
    "hidden_dim": 1024,
    "max_seq_len": 3456,
    "model_stride": 3,
    "lr": 5e-5,
    "base_lr": 1e-6,
    "weight_decay": 0.05,
    "freeze_epochs": 50,
    "eval_every": 1,
    "num_workers": 8,
    "train_ratio": 0.8,
    "seed": 42,
    "glucose_scale": 1.0,
    "dropout": 0.1,
    "device": "cuda:0",
    "event_cols": [
        "Diet",
        "Calories",
        "Carbohydrates",
        "Protein",
        "Fat",
        "Fiber",
    ],
    "use_cnn": True,
    "use_transformer": True,
    "use_temporal": False,
    "use_positional": True,
    "normalize_continuous_events": False,
    "save_best_checkpoint": True,
    "save_final_checkpoint": True,
    "export_predictions": True,
    "prediction_split": "val",
    "prediction_patch_chunk_size": 0,
}


DEFAULT_TASK_NAMES = ["CGMacros", "HPP", "zhao_simple", "zhao_full"]

ZHAO_SIMPLE_EVENT_COLS = [
    "Non-insulin hypoglycemic agents",
    "Insulin",
    "Diet",
    "Exercise",
    "Calories",
    "Carbohydrates",
    "Protein",
    "Fat",
    "Fiber",
]

ZHAO_FULL_EVENT_COLS = [
    "DPP-4 inhibitors",
    "GLP-1 receptor agonists",
    "Glinides",
    "Glucokinase activators",
    "Metformin",
    "PPAR pan-agonists",
    "SGLT-2 inhibitors",
    "Sulfonylureas",
    "Thiazolidinediones",
    "α-Glucosidase inhibitors",
    "Insulin",
    "Insulin pump",
    "Intermediate-acting insulin",
    "Long-acting insulin",
    "Premixed insulin",
    "Short-acting insulin",
    "Ultra-rapid-acting insulin",
    "Exercise",
    "Diet",
    "Calories",
    "Carbohydrates",
    "Protein",
    "Fat",
    "Fiber",
    "brand_novolin_r",
    "brand_novolin_r_iu",
    "brand_insulin_aspart_70_30",
    "brand_insulin_aspart_70_30_iu",
    "brand_insulin_degludec",
    "brand_insulin_degludec_iu",
    "brand_insulin_detemir",
    "brand_insulin_detemir_iu",
    "brand_novolin_30r",
    "brand_novolin_30r_iu",
    "brand_humulin_r",
    "brand_humulin_r_iu",
    "brand_humulin_70_30",
    "brand_humulin_70_30_iu",
    "brand_gansulin_r",
    "brand_gansulin_r_iu",
    "brand_insulin_glargine",
    "brand_insulin_glargine_iu",
    "brand_insulin_glulisine",
    "brand_insulin_glulisine_iu",
    "brand_insulin_aspart",
    "brand_insulin_aspart_iu",
    "brand_gansulin_40r",
    "brand_gansulin_40r_iu",
    "brand_novolin_50r",
    "brand_novolin_50r_iu",
    "brand_scilin_m30",
    "brand_scilin_m30_iu",
]


def build_default_tasks(repo_root: Path):
    data_root = repo_root / "data_event"
    return [
        {
            "name": "CGMacros",
            "data_csv": data_root
            / "CGMacros"
            / "CGMacros_data_withevent_simple_5min_3456_libre_mmol_from00.csv",
            "onehot_csv": data_root
            / "CGMacros"
            / "CGMacros_OneHot_all_simple_5min_3456_libre_mmol_from00.csv",
        },
        {
            "name": "HPP",
            "data_csv": data_root
            / "HPP"
            / "participant_cgm_data_withevent_full_5min_3456_from00.csv",
            "onehot_csv": data_root / "HPP" / "participant_diet_events_only_5min.csv",
        },
        {
            "name": "zhao_simple",
            "data_csv": data_root
            / "zhao"
            / "zhao_data_withevent_simple_5min_3456_from00.csv",
            "onehot_csv": data_root
            / "zhao"
            / "zhao_OneHot_all_simple_5min_3456_from00.csv",
            "event_cols": ZHAO_SIMPLE_EVENT_COLS,
        },
        {
            "name": "zhao_full",
            "data_csv": data_root
            / "zhao"
            / "zhao_data_withevent_full_5min_3456_from00.csv",
            "onehot_csv": data_root
            / "zhao"
            / "zhao_OneHot_all_full_5min_3456_from00.csv",
            "event_cols": ZHAO_FULL_EVENT_COLS,
        },
    ]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run all EventFinetune tasks and distribute them across multiple GPUs."
    )
    parser.add_argument(
        "--gpus",
        type=str,
        required=True,
        help="Comma-separated GPU ids, such as 0,1,2.",
    )
    parser.add_argument(
        "--template_config",
        type=str,
        default=None,
        help="Optional JSON config used as the common base for all tasks.",
    )
    parser.add_argument(
        "--pretrain_ckpt",
        type=str,
        default=None,
        help="Override pretrain checkpoint path for all tasks.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="data_event/event_finetune_outputs",
        help="Base output directory. Each task will use a separate subdirectory.",
    )
    parser.add_argument(
        "--config_out_dir",
        type=str,
        default="configs/event_finetune_tasks",
        help="Directory used to persist generated per-task JSON configs.",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="data_event/event_finetune_outputs/_scheduler_logs",
        help="Directory used to persist per-task worker logs.",
    )
    parser.add_argument(
        "--run_name_prefix",
        type=str,
        default="event_finetune",
        help="Prefix used to build task-specific run_name values.",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument(
        "--prediction_split",
        type=str,
        choices=["train", "val", "all"],
        default=None,
    )
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="*",
        default=None,
        help="Optional subset of tasks. Default runs all tasks.",
    )
    parser.add_argument(
        "--poll_interval",
        type=int,
        default=5,
        help="Seconds between scheduler polls.",
    )
    return parser.parse_args()


def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_absolute(path_str, base_dir: Path):
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return str(path.resolve())
    return str((base_dir / path).resolve())


def load_base_config(args):
    base_config = deepcopy(DEFAULT_BASE_CONFIG)
    template_path = None
    if args.template_config:
        template_path = Path(args.template_config).expanduser().resolve()
        template_config = load_json(template_path)
        base_config.update(template_config)

    if template_path is not None:
        base_dir = template_path.parent
        if base_config.get("pretrain_ckpt"):
            base_config["pretrain_ckpt"] = resolve_absolute(
                base_config["pretrain_ckpt"], base_dir
            )

    if args.pretrain_ckpt:
        base_config["pretrain_ckpt"] = str(
            Path(args.pretrain_ckpt).expanduser().resolve()
        )

    if not base_config.get("pretrain_ckpt"):
        raise ValueError(
            "pretrain_ckpt is required. Pass --pretrain_ckpt or provide it in --template_config."
        )

    base_config["out_dir"] = str(Path(args.out_dir).expanduser())
    if args.epochs is not None:
        base_config["epochs"] = int(args.epochs)
    if args.batch_size is not None:
        base_config["batch_size"] = int(args.batch_size)
    if args.num_workers is not None:
        base_config["num_workers"] = int(args.num_workers)
    if args.prediction_split is not None:
        base_config["prediction_split"] = args.prediction_split

    return base_config


def select_tasks(all_tasks, selected_names):
    task_map = {task["name"]: task for task in all_tasks}
    if not selected_names:
        selected_names = DEFAULT_TASK_NAMES

    missing = [name for name in selected_names if name not in task_map]
    if missing:
        available = ", ".join(task_map.keys())
        raise ValueError(f"Unknown tasks: {missing}. Available: {available}")
    return [task_map[name] for name in selected_names]


def create_task_configs(
    base_config,
    tasks,
    run_name_prefix,
    output_root: Path,
    config_out_dir: Path,
):
    output_root.mkdir(parents=True, exist_ok=True)
    config_out_dir.mkdir(parents=True, exist_ok=True)

    task_configs = []
    for task in tasks:
        config = deepcopy(base_config)
        config["data_csv"] = str(task["data_csv"].resolve())
        config["onehot_csv"] = str(task["onehot_csv"].resolve())
        config["run_name"] = f"{run_name_prefix}_{task['name']}"
        config["out_dir"] = str((output_root / task["name"]).resolve())
        if "event_cols" in task:
            config["event_cols"] = list(task["event_cols"])

        config_path = config_out_dir / f"{task['name']}.json"
        config_path.write_text(
            json.dumps(config, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        task_configs.append({"name": task["name"], "config_path": config_path})

    manifest = {
        "script": "EventFinetune_main.py",
        "output_root": str(output_root),
        "tasks": [
            {"name": item["name"], "config_path": str(item["config_path"])}
            for item in task_configs
        ],
    }
    manifest_path = config_out_dir / "_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return task_configs


def close_log_handle(item):
    log_handle = item.get("log_handle")
    if log_handle is not None and not log_handle.closed:
        log_handle.close()


def stop_running_tasks(running, grace_seconds=5.0):
    alive_items = [item for item in running if item["process"].poll() is None]
    for item in alive_items:
        try:
            item["process"].terminate()
        except ProcessLookupError:
            pass

    deadline = time.time() + grace_seconds
    while time.time() < deadline:
        if all(item["process"].poll() is not None for item in alive_items):
            break
        time.sleep(0.2)

    for item in alive_items:
        if item["process"].poll() is None:
            try:
                item["process"].kill()
            except ProcessLookupError:
                pass

    for item in running:
        close_log_handle(item)


def write_runtime_files(config_out_dir: Path, log_dir: Path):
    runtime = {
        "pid": os.getpid(),
        "pgid": os.getpgrp(),
        "log_dir": str(log_dir),
        "written_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (config_out_dir / "_scheduler.pid").write_text(
        f"{runtime['pid']}\n", encoding="utf-8"
    )
    (config_out_dir / "_scheduler.pgid").write_text(
        f"{runtime['pgid']}\n", encoding="utf-8"
    )
    (config_out_dir / "_scheduler_runtime.json").write_text(
        json.dumps(runtime, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return runtime


def main():
    args = parse_args()
    script_path = Path(__file__).with_name("EventFinetune_main.py").resolve()
    repo_root = script_path.parent

    gpu_ids = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if not gpu_ids:
        raise ValueError("--gpus cannot be empty.")

    all_tasks = build_default_tasks(repo_root)
    tasks = select_tasks(all_tasks, args.tasks)
    base_config = load_base_config(args)
    output_root = Path(args.out_dir).expanduser().resolve()
    config_out_dir = Path(args.config_out_dir).expanduser()
    if not config_out_dir.is_absolute():
        config_out_dir = (repo_root / config_out_dir).resolve()
    config_out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.log_dir).expanduser()
    if not log_dir.is_absolute():
        log_dir = (repo_root / log_dir).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)

    runtime = write_runtime_files(config_out_dir, log_dir)

    task_configs = create_task_configs(
        base_config=base_config,
        tasks=tasks,
        run_name_prefix=args.run_name_prefix,
        output_root=output_root,
        config_out_dir=config_out_dir,
    )

    print(f"[SCHEDULER] Saved task configs to {config_out_dir}")
    print(f"[SCHEDULER] Worker logs will be written to {log_dir}")
    print(
        f"[SCHEDULER] Runtime files: pid={config_out_dir / '_scheduler.pid'} "
        f"pgid={config_out_dir / '_scheduler.pgid'}"
    )
    print(f"[SCHEDULER] Running with pid={runtime['pid']} pgid={runtime['pgid']}")

    queue = list(task_configs)
    free_gpus = list(gpu_ids)
    running = []
    failures = []
    stop_requested = False

    def handle_stop_signal(signum, _frame):
        nonlocal stop_requested
        signal_name = signal.Signals(signum).name
        print(f"[SCHEDULER] Received {signal_name}. Stopping running tasks...")
        stop_requested = True
        stop_running_tasks(running)

    previous_handlers = {
        sig: signal.signal(sig, handle_stop_signal)
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
    }

    try:
        while queue or running:
            if stop_requested:
                return

            while queue and free_gpus:
                gpu_id = free_gpus.pop(0)
                task = queue.pop(0)

                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
                env["PYTHONUNBUFFERED"] = "1"
                existing_pythonpath = env.get("PYTHONPATH", "")
                env["PYTHONPATH"] = (
                    str(repo_root)
                    if not existing_pythonpath
                    else str(repo_root) + os.pathsep + existing_pythonpath
                )
                cmd = [
                    sys.executable,
                    str(script_path),
                    "--config",
                    str(task["config_path"]),
                    "--device",
                    "cuda:0",
                ]
                log_path = log_dir / f"{task['name']}.log"
                log_handle = log_path.open("a", encoding="utf-8")
                proc = subprocess.Popen(
                    cmd,
                    env=env,
                    cwd=str(repo_root),
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                )
                running.append(
                    {
                        "gpu": gpu_id,
                        "task": task["name"],
                        "process": proc,
                        "command": cmd,
                        "log_path": log_path,
                        "log_handle": log_handle,
                    }
                )
                print(
                    f"[SCHEDULER] Started task={task['name']} on gpu={gpu_id} "
                    f"(pid={proc.pid}, log={log_path})"
                )

            if running:
                time.sleep(args.poll_interval)

            still_running = []
            for item in running:
                return_code = item["process"].poll()
                if return_code is None:
                    still_running.append(item)
                    continue

                free_gpus.append(item["gpu"])
                item["log_handle"].close()
                if return_code == 0:
                    print(
                        f"[SCHEDULER] Finished task={item['task']} on gpu={item['gpu']} "
                        f"(log={item['log_path']})"
                    )
                else:
                    failures.append(
                        {
                            "task": item["task"],
                            "gpu": item["gpu"],
                            "return_code": return_code,
                            "log_path": str(item["log_path"]),
                        }
                    )
                    print(
                        f"[SCHEDULER] Failed task={item['task']} on gpu={item['gpu']} "
                        f"(code={return_code}, log={item['log_path']})"
                    )

            running = still_running
            free_gpus.sort(key=lambda value: int(value) if value.isdigit() else value)
    except KeyboardInterrupt:
        print("[SCHEDULER] Interrupted. Stopping running tasks...")
        stop_running_tasks(running)
        return
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
        for item in running:
            close_log_handle(item)

    if failures:
        raise SystemExit(
            "[SCHEDULER] Some tasks failed: "
            + ", ".join(
                f"{item['task']}@gpu{item['gpu']} code={item['return_code']} log={item['log_path']}"
                for item in failures
            )
        )

    print("[SCHEDULER] All tasks completed successfully.")


if __name__ == "__main__":
    main()
