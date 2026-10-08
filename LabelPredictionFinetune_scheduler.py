import argparse
import copy
import importlib.util
import json
import queue
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch
import torch.multiprocessing as mp

REPO_ROOT = Path(__file__).resolve().parent
CODE_ROOT = REPO_ROOT

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from LabelPredictionFinetune import lab_cv_runner as runner


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Queue multiple lab training configs and dispatch target tasks to idle GPUs."
        )
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Scheduler JSON config file.",
    )
    return parser


def save_json(path: Path, payload: dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_scheduler_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Scheduler config not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError("Scheduler config must be a JSON object.")
    return raw


def resolve_path(path_str: str, base_dir: Path) -> Path:
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (base_dir / path).resolve()


def build_job_specs(raw_jobs: list[Any], base_dir: Path) -> list[dict[str, Any]]:
    jobs = []
    for index, item in enumerate(raw_jobs, start=1):
        if isinstance(item, str):
            config_path = resolve_path(item, base_dir)
            jobs.append(
                {
                    "name": config_path.stem,
                    "config": config_path,
                }
            )
            continue

        if not isinstance(item, dict):
            raise ValueError(f"Unsupported job item at index {index}: {item!r}")

        config_value = item.get("config")
        if not config_value:
            raise ValueError(f"Job at index {index} is missing 'config'.")

        config_path = resolve_path(str(config_value), base_dir)
        jobs.append(
            {
                "name": str(item.get("name") or config_path.stem),
                "config": config_path,
            }
        )
    return jobs


def configure_runner_from_script(runner_script: Path):
    module_name = f"_cgmfoundation_runner_{runner.sanitize_name(str(runner_script))}"
    module = sys.modules.get(module_name)
    if module is None:
        spec = importlib.util.spec_from_file_location(module_name, str(runner_script))
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load runner script: {runner_script}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

    if hasattr(module, "configure_runner"):
        module.configure_runner()
    return module


def build_default_runner_args():
    return runner.build_parser().parse_args([])


def apply_flat_config(default_args, flat_config: dict[str, Any], config_path: Path):
    args = copy.deepcopy(default_args)
    args.config = str(config_path)
    for key, value in flat_config.items():
        if hasattr(args, key):
            setattr(args, key, value)
    return args


def prepare_job_bundle(
    job: dict[str, Any],
    default_args,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    raw_config = runner.load_json(job["config"])
    if not isinstance(raw_config, dict):
        raise ValueError(f"Runner config must be a JSON object: {job['config']}")

    flat_config = runner.flatten_config(raw_config, job["config"].parent)
    args = apply_flat_config(default_args, flat_config, job["config"])

    if not args.dataset:
        raise ValueError(f"dataset is required in runner config: {job['config']}")
    if args.prepare_only:
        raise ValueError(
            f"Scheduler expects training configs, but prepare_only=true in: {job['config']}"
        )

    cgm_df, labels_df = runner.load_dataset_frames(args.dataset)

    if args.targets:
        target_specs = runner.parse_target_specs(args.targets, labels_df)
    elif args.prepare_all_targets:
        target_specs = runner.parse_target_specs(labels_df.columns.tolist(), labels_df)
    else:
        raise ValueError(f"Training requires targets in config: {job['config']}")

    if args.ckpt_path:
        ckpt_path = Path(args.ckpt_path).expanduser().resolve()
    elif args.event_pretrain_task:
        ckpt_path = runner.resolve_event_pretrain_checkpoint(
            task_name=args.event_pretrain_task,
            config_dir=Path(args.event_config_dir).expanduser().resolve(),
            out_root=Path(args.event_out_root).expanduser().resolve(),
        )
    else:
        raise ValueError(
            f"Training requires ckpt_path or event_pretrain_task in: {job['config']}"
        )

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    payload, _ = runner.load_checkpoint_payload(ckpt_path)
    ckpt_hparams = runner.extract_base_model_hparams(payload)
    model_config = runner.resolve_model_config(args, ckpt_hparams)
    pre_bundle = runner.pre_process_all_cgm(
        cgm_df, max_seq_len=model_config["max_seq_len"]
    )

    split_root = Path(args.split_root).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()
    model_name = runner.infer_model_name(args)
    run_root = out_root / model_name / args.dataset
    run_root.mkdir(parents=True, exist_ok=True)

    run_config = {
        "dataset": args.dataset,
        "targets": target_specs,
        "config_path": args.config,
        "checkpoint": str(ckpt_path),
        "event_pretrain_task": args.event_pretrain_task,
        "model_name": model_name,
        "split_root": str(split_root),
        "scheduler_mode": "global_target_queue",
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "layer_decay": args.layer_decay,
            "freeze_base": args.freeze_base,
            "unfreeze_after": args.unfreeze_after,
            "patience": args.patience,
            "min_samples": args.min_samples,
            "folds": args.folds,
            "wait_for_idle_gpus": args.wait_for_idle_gpus,
            "idle_min_memory_gb": args.idle_min_memory_gb,
            "idle_max_utilization": args.idle_max_utilization,
            "idle_poll_seconds": args.idle_poll_seconds,
        },
        "model_config": model_config,
    }
    runner.save_json(run_root / "_resolved_config.json", run_config)

    runtime_config = {
        "dataset": args.dataset,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "layer_decay": args.layer_decay,
        "freeze_base": args.freeze_base,
        "unfreeze_after": args.unfreeze_after,
        "patience": args.patience,
        "disable_compile": args.disable_compile,
        "selected_folds": set(args.folds) if args.folds else None,
    }

    job_key = str(run_root)
    task_entries = []
    for spec in target_specs:
        target = spec["target"]
        mode = spec["mode"]
        task_df, label_map = runner.build_task_frame(
            labels_df=labels_df,
            spec=spec,
            min_samples=args.min_samples,
        )
        if task_df is None:
            print(
                f"[SKIP] dataset={args.dataset} target={target}: "
                f"valid sample count < {args.min_samples} or only one class."
            )
            continue

        task_name = str(spec.get("task_name") or runner.build_task_name(target, mode))
        split_payload = runner.ensure_target_split(
            dataset_name=args.dataset,
            task_name=task_name,
            target_col=target,
            task_type=mode,
            task_df=task_df,
            split_root=split_root,
            n_splits=args.n_splits,
            seed=args.seed,
            force_regenerate=args.force_regenerate_splits,
        )
        split_task_root = (
            split_root
            / args.dataset
            / task_name
            / f"kfold_{args.n_splits}_seed_{args.seed}"
        )
        split_manifest_path = split_task_root / "split_manifest.json"
        print(
            f"[TARGET_SPLIT:{split_payload.get('_status', 'ok').upper()}] "
            f"{args.dataset}/{task_name} -> {split_payload['num_samples']} samples, "
            f"file={split_manifest_path}"
        )
        runner.save_json(
            split_task_root / "target_info.json",
            {
                "dataset": args.dataset,
                "target": target,
                "task_name": task_name,
                "task_type": mode,
                "num_samples": int(len(task_df)),
                "num_classes": len(label_map) if label_map is not None else None,
                "all_samples_csv": split_payload["all_samples_csv"],
                "target_spec": spec,
            },
        )
        if label_map is not None:
            runner.save_json(
                split_task_root / "label_mapping.json",
                {
                    "target": target,
                    "mapping": {
                        str(key): int(value) for key, value in label_map.items()
                    },
                },
            )

        task_entries.append(
            {
                "job_key": job_key,
                "task_name": task_name,
                "task_bundle": {
                    "dataset": args.dataset,
                    "target": target,
                    "mode": mode,
                    "task_name": task_name,
                    "task_df": task_df,
                    "label_map": label_map,
                    "split_payload": split_payload,
                    "split_manifest_path": str(split_manifest_path),
                    "spec": spec,
                },
            }
        )

    job_context = {
        "job_key": job_key,
        "job_name": job["name"],
        "dataset": args.dataset,
        "model_name": model_name,
        "run_root": run_root,
        "pre_bundle": pre_bundle,
        "model_config": model_config,
        "ckpt_path": ckpt_path,
        "runtime_config": runtime_config,
        "source_config": str(job["config"]),
    }
    return job_context, task_entries


def gpu_worker(
    task_queue,
    result_queue,
    device_str: str,
    runner_script_str: str,
    job_contexts: dict[str, dict[str, Any]],
    log_dir_str: str,
):
    log_path = Path(log_dir_str) / f"gpu_{device_str.replace(':', '_')}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("a", encoding="utf-8", buffering=1)
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = handle
    sys.stderr = handle

    try:
        configure_runner_from_script(Path(runner_script_str))
        runner.set_gpu_flags()
        device = torch.device(device_str)
        if device.type == "cuda":
            torch.cuda.set_device(device)

        print(
            f"\n{'=' * 20} Worker {device_str} Started {time.strftime('%Y-%m-%d %H:%M:%S')} {'=' * 20}",
            flush=True,
        )
        result_queue.put(
            {
                "type": "worker_started",
                "device": device_str,
                "log_path": str(log_path),
            }
        )

        while True:
            try:
                task_entry = task_queue.get_nowait()
            except queue.Empty:
                break
            except Exception:
                break

            job_key = task_entry["job_key"]
            task_bundle = task_entry["task_bundle"]
            context = job_contexts[job_key]

            print(
                f"\n>>> Task: {context['dataset']}/{task_bundle['task_name']} on {device_str}",
                flush=True,
            )
            result = runner.run_task_bundle(
                task_bundle=task_bundle,
                run_root=Path(context["run_root"]),
                pre_bundle=context["pre_bundle"],
                model_config=context["model_config"],
                ckpt_path=Path(context["ckpt_path"]),
                runtime_config=context["runtime_config"],
                device_str=device_str,
            )
            result_queue.put(
                {
                    "type": "task_result",
                    "job_key": job_key,
                    "device": device_str,
                    "dataset": context["dataset"],
                    "task_name": task_bundle["task_name"],
                    **result,
                }
            )

        print(f"\n{'=' * 20} Worker {device_str} All Done {'=' * 20}", flush=True)
    except Exception:
        result_queue.put(
            {
                "type": "worker_error",
                "device": device_str,
                "log_path": str(log_path),
                "traceback": traceback.format_exc(),
            }
        )
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        handle.close()


def main():
    args = build_parser().parse_args()
    config_path = Path(args.config).expanduser().resolve()
    raw_config = load_scheduler_config(config_path)
    base_dir = config_path.parent

    runner_script = resolve_path(
        str(raw_config.get("runner_script", "../../LabelPredictionFinetune_CV_main.py")),
        base_dir,
    )
    if not runner_script.exists():
        raise FileNotFoundError(f"Runner script not found: {runner_script}")

    configure_runner_from_script(runner_script)

    raw_jobs = raw_config.get("jobs")
    if not isinstance(raw_jobs, list) or not raw_jobs:
        raise ValueError("Scheduler config must contain a non-empty 'jobs' list.")

    dispatch_mode = str(raw_config.get("dispatch_mode", "target")).strip().lower()
    if dispatch_mode != "target":
        raise ValueError("This scheduler now only supports dispatch_mode='target'.")

    devices = runner.normalize_device_list(
        raw_config.get("devices"),
        raw_config.get("device", "cuda:0"),
    )
    if not devices:
        raise ValueError("At least one GPU device must be configured.")

    wait_for_idle_gpus = bool(raw_config.get("wait_for_idle_gpus", True))
    idle_min_memory_gb = float(raw_config.get("idle_min_memory_gb", 8.0))
    idle_max_utilization = float(raw_config.get("idle_max_utilization", 30.0))
    idle_poll_seconds = float(raw_config.get("idle_poll_seconds", 60.0))

    log_dir = resolve_path(
        str(
            raw_config.get("log_dir", "../../outputs/CGMfoundation_lab_scheduler/logs")
        ),
        base_dir,
    )
    log_dir.mkdir(parents=True, exist_ok=True)

    default_runner_args = build_default_runner_args()
    jobs = build_job_specs(raw_jobs, base_dir)

    job_contexts: dict[str, dict[str, Any]] = {}
    task_entries: list[dict[str, Any]] = []
    for job in jobs:
        context, prepared_tasks = prepare_job_bundle(job, default_runner_args)
        if context["job_key"] in job_contexts:
            raise ValueError(
                f"Duplicate run_root detected between configs: {context['run_root']}"
            )
        job_contexts[context["job_key"]] = context
        task_entries.extend(prepared_tasks)

    summary_rows = {
        job_key: {"task_summaries": [], "fold_metrics": []}
        for job_key in job_contexts.keys()
    }

    manifest = {
        "runner_script": str(runner_script),
        "dispatch_mode": "target",
        "devices": devices,
        "jobs": [
            {
                "job_name": context["job_name"],
                "dataset": context["dataset"],
                "model_name": context["model_name"],
                "run_root": str(context["run_root"]),
                "source_config": context["source_config"],
            }
            for context in job_contexts.values()
        ],
        "num_tasks": len(task_entries),
    }
    save_json(
        log_dir / f"queue_manifest_{time.strftime('%Y%m%d_%H%M%S')}.json", manifest
    )

    if not task_entries:
        print(
            "[QUEUE_DONE] no runnable target tasks were produced from the provided configs."
        )
        return

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    ctx = mp.get_context("spawn")
    task_queue = ctx.Queue()
    result_queue = ctx.Queue()

    for task_entry in task_entries:
        task_queue.put(task_entry)

    pending_devices = list(devices)
    workers: dict[str, mp.Process] = {}
    expected_results = len(task_entries)
    received_results = 0
    worker_errors: list[dict[str, Any]] = []
    completed_tasks: list[dict[str, Any]] = []

    print(f"[QUEUE_START] tasks={expected_results} devices={devices}")

    while received_results < expected_results or workers:
        for device_str in pending_devices.copy():
            if received_results >= expected_results:
                break

            if wait_for_idle_gpus:
                idle, status = runner.is_device_idle(
                    device=device_str,
                    min_free_memory_gb=idle_min_memory_gb,
                    max_utilization=idle_max_utilization,
                )
                if not idle:
                    print(
                        f"[WAIT_GPU] {device_str} free={status.get('free_memory_gb')}GB "
                        f"util={status.get('utilization')}% "
                        f"(need free>={idle_min_memory_gb}GB util<={idle_max_utilization}%)"
                    )
                    continue

            worker = ctx.Process(
                target=gpu_worker,
                args=(
                    task_queue,
                    result_queue,
                    device_str,
                    str(runner_script),
                    job_contexts,
                    str(log_dir),
                ),
            )
            worker.start()
            workers[device_str] = worker
            pending_devices.remove(device_str)
            print(f"[LAUNCH_GPU] device={device_str} pid={worker.pid}")

        try:
            message = result_queue.get(timeout=max(idle_poll_seconds, 1.0))
        except queue.Empty:
            if pending_devices:
                continue
            message = None

        if message is not None:
            if message["type"] == "task_result":
                received_results += 1
                job_key = message["job_key"]
                runner.collect_aggregate_rows(
                    message,
                    summary_rows[job_key]["task_summaries"],
                    summary_rows[job_key]["fold_metrics"],
                )
                completed_tasks.append(
                    {
                        "job_key": job_key,
                        "dataset": message["dataset"],
                        "task_name": message["task_name"],
                        "device": message["device"],
                    }
                )
            elif message["type"] == "worker_started":
                print(
                    f"[WORKER_STARTED] device={message['device']} log={message['log_path']}"
                )
            elif message["type"] == "worker_error":
                worker_errors.append(message)
                break

        finished_devices = []
        for device_str, worker in workers.items():
            if not worker.is_alive():
                worker.join()
                finished_devices.append(device_str)
        for device_str in finished_devices:
            workers.pop(device_str, None)

        if pending_devices and wait_for_idle_gpus:
            time.sleep(max(idle_poll_seconds, 1.0))

    for worker in workers.values():
        worker.join()

    if worker_errors:
        error = worker_errors[0]
        raise RuntimeError(
            f"Worker failed on {error['device']}. Log: {error['log_path']}\n{error['traceback']}"
        )

    aggregate_reports = []
    for job_key, context in job_contexts.items():
        runner.write_run_aggregates(
            run_root=Path(context["run_root"]),
            all_task_summaries=summary_rows[job_key]["task_summaries"],
            all_fold_metrics=summary_rows[job_key]["fold_metrics"],
        )
        aggregate_reports.append(
            {
                "job_name": context["job_name"],
                "dataset": context["dataset"],
                "model_name": context["model_name"],
                "run_root": str(context["run_root"]),
                "num_task_rows": len(summary_rows[job_key]["task_summaries"]),
                "num_fold_rows": len(summary_rows[job_key]["fold_metrics"]),
                "source_config": context["source_config"],
            }
        )
        print(
            f"[AGGREGATE_DONE] run_root={context['run_root']} "
            f"tasks={len(summary_rows[job_key]['task_summaries'])} "
            f"folds={len(summary_rows[job_key]['fold_metrics'])}"
        )

    summary = {
        "runner_script": str(runner_script),
        "dispatch_mode": "target",
        "devices": devices,
        "completed_tasks": completed_tasks,
        "aggregate_reports": aggregate_reports,
    }
    summary_path = log_dir / f"queue_summary_{time.strftime('%Y%m%d_%H%M%S')}.json"
    save_json(summary_path, summary)
    print(f"[QUEUE_DONE] summary={summary_path}")


if __name__ == "__main__":
    main()
