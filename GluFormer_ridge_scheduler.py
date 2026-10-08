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
            "Queue multiple GluFormer ridge configs and dispatch dataset jobs to idle GPUs."
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
                    "overrides": {},
                    "override_base_dir": base_dir,
                }
            )
            continue

        if not isinstance(item, dict):
            raise ValueError(f"Unsupported job item at index {index}: {item!r}")

        config_value = item.get("config")
        if not config_value:
            raise ValueError(f"Job at index {index} is missing 'config'.")

        config_path = resolve_path(str(config_value), base_dir)
        overrides = item.get("overrides", {})
        if overrides is None:
            overrides = {}
        if not isinstance(overrides, dict):
            raise ValueError(f"Job overrides must be a JSON object: {item!r}")

        jobs.append(
            {
                "name": str(item.get("name") or config_path.stem),
                "config": config_path,
                "overrides": overrides,
                "override_base_dir": base_dir,
            }
        )
    return jobs


def configure_runner_from_script(runner_script: Path):
    module_name = f"_gluformer_lab_ridge_runner_{runner.sanitize_name(str(runner_script))}"
    module = sys.modules.get(module_name)
    if module is None:
        spec = importlib.util.spec_from_file_location(module_name, str(runner_script))
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load runner script: {runner_script}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    return module


def build_default_runner_args(runner_module):
    return runner_module.build_parser().parse_args([])


def apply_flat_config(default_args, flat_config: dict[str, Any], config_path: Path):
    args = copy.deepcopy(default_args)
    args.config = str(config_path)
    for key, value in flat_config.items():
        if hasattr(args, key):
            setattr(args, key, value)
    return args


def flatten_scheduler_overrides(
    raw_overrides: dict[str, Any],
    runner_module,
    base_dir: Path,
) -> dict[str, Any]:
    if not raw_overrides:
        return {}
    if hasattr(runner_module, "flatten_config"):
        return runner_module.flatten_config(raw_overrides, base_dir)
    return raw_overrides


def prepare_job_bundle(
    job: dict[str, Any],
    default_args,
    runner_module,
    common_overrides: dict[str, Any],
) -> dict[str, Any]:
    raw_config = runner_module.load_json(job["config"])
    if not isinstance(raw_config, dict):
        raise ValueError(f"Runner config must be a JSON object: {job['config']}")

    flat_config = runner_module.flatten_config(raw_config, job["config"].parent)
    job_overrides = flatten_scheduler_overrides(
        job["overrides"],
        runner_module,
        job["override_base_dir"],
    )

    merged_config = dict(flat_config)
    merged_config.update(common_overrides)
    merged_config.update(job_overrides)

    args = apply_flat_config(default_args, merged_config, job["config"])
    if not args.dataset:
        raise ValueError(f"dataset is required in runner config: {job['config']}")
    if not args.ckpt_path:
        raise ValueError(
            f"ckpt_path is required after overrides are applied: {job['config']}"
        )
    if not args.prepare_only and not args.prepare_all_targets and not args.targets:
        raise ValueError(
            "targets are required unless prepare_only=true or prepare_all_targets=true: "
            f"{job['config']}"
        )

    ckpt_path = Path(args.ckpt_path).expanduser().resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    args.ckpt_path = str(ckpt_path)

    model_name = runner_module.infer_model_name(args)
    run_root = Path(args.out_root).expanduser().resolve() / model_name / args.dataset

    return {
        "job_name": job["name"],
        "source_config": str(job["config"]),
        "dataset": args.dataset,
        "model_name": model_name,
        "run_root": str(run_root),
        "args": vars(args).copy(),
    }


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
        runner_module = configure_runner_from_script(Path(runner_script_str))
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
            context = job_contexts[job_key]
            args = argparse.Namespace(**copy.deepcopy(context["args"]))
            args.device = device_str

            print(
                f"\n>>> Job: {context['dataset']} on {device_str}",
                flush=True,
            )
            result = runner_module.run_from_args(args)
            result_queue.put(
                {
                    "type": "job_result",
                    "job_key": job_key,
                    "job_name": context["job_name"],
                    "device": device_str,
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
        str(raw_config.get("runner_script", "../../GluFormer_ridge_main.py")),
        base_dir,
    )
    if not runner_script.exists():
        raise FileNotFoundError(f"Runner script not found: {runner_script}")

    runner_module = configure_runner_from_script(runner_script)

    raw_jobs = raw_config.get("jobs")
    if not isinstance(raw_jobs, list) or not raw_jobs:
        raise ValueError("Scheduler config must contain a non-empty 'jobs' list.")

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
            raw_config.get(
                "log_dir",
                "../../outputs/gluformer_lab_ridge_scheduler/logs",
            )
        ),
        base_dir,
    )
    log_dir.mkdir(parents=True, exist_ok=True)
    scheduler_out_root = log_dir.parent
    scheduler_out_root.mkdir(parents=True, exist_ok=True)

    default_runner_args = build_default_runner_args(runner_module)
    jobs = build_job_specs(raw_jobs, base_dir)
    common_overrides = flatten_scheduler_overrides(
        raw_config.get("common_overrides") or raw_config.get("common_args") or {},
        runner_module,
        base_dir,
    )

    job_contexts: dict[str, dict[str, Any]] = {}
    task_entries: list[dict[str, Any]] = []
    for job in jobs:
        context = prepare_job_bundle(
            job=job,
            default_args=default_runner_args,
            runner_module=runner_module,
            common_overrides=common_overrides,
        )
        job_key = context["run_root"]
        if job_key in job_contexts:
            raise ValueError(f"Duplicate run_root detected between jobs: {job_key}")
        job_contexts[job_key] = context
        task_entries.append({"job_key": job_key})

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    manifest = {
        "runner_script": str(runner_script),
        "devices": devices,
        "wait_for_idle_gpus": wait_for_idle_gpus,
        "idle_min_memory_gb": idle_min_memory_gb,
        "idle_max_utilization": idle_max_utilization,
        "idle_poll_seconds": idle_poll_seconds,
        "jobs": [
            {
                "job_name": context["job_name"],
                "dataset": context["dataset"],
                "model_name": context["model_name"],
                "run_root": context["run_root"],
                "source_config": context["source_config"],
            }
            for context in job_contexts.values()
        ],
        "num_jobs": len(task_entries),
    }
    manifest_path = log_dir / f"queue_manifest_{timestamp}.json"
    save_json(manifest_path, manifest)

    if not task_entries:
        print("[QUEUE_DONE] no runnable dataset jobs were produced from the provided configs.")
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
    completed_jobs: list[dict[str, Any]] = []
    all_task_summaries: list[dict[str, Any]] = []
    all_fold_metrics: list[dict[str, Any]] = []

    print(f"[QUEUE_START] jobs={expected_results} devices={devices}")

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
            if message["type"] == "job_result":
                received_results += 1
                completed_jobs.append(
                    {
                        "job_name": message["job_name"],
                        "dataset": message["dataset"],
                        "model_name": message["model_name"],
                        "device": message["device"],
                        "run_root": message["run_root"],
                        "num_tasks": int(message.get("num_tasks", 0)),
                        "num_fold_rows": int(message.get("num_fold_rows", 0)),
                        "resolved_config_path": message.get("resolved_config_path"),
                        "embedding_cache_path": message.get("embedding_cache_path"),
                        "embedding_cache_meta_path": message.get(
                            "embedding_cache_meta_path"
                        ),
                    }
                )
                all_task_summaries.extend(message.get("task_summaries", []))
                all_fold_metrics.extend(message.get("fold_metrics_rows", []))
                print(
                    f"[JOB_DONE] dataset={message['dataset']} device={message['device']} "
                    f"run_root={message['run_root']}"
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

    task_summary_path = None
    if all_task_summaries:
        task_summary_path = (
            scheduler_out_root / f"all_jobs_task_summary_{timestamp}.json"
        )
        save_json(task_summary_path, {"rows": all_task_summaries})

    fold_metrics_path = None
    if all_fold_metrics:
        fold_metrics_path = (
            scheduler_out_root / f"all_jobs_fold_metrics_{timestamp}.json"
        )
        save_json(fold_metrics_path, {"rows": all_fold_metrics})

    summary = {
        "runner_script": str(runner_script),
        "devices": devices,
        "manifest_path": str(manifest_path),
        "scheduler_out_root": str(scheduler_out_root),
        "log_dir": str(log_dir),
        "all_jobs_task_summary_json": str(task_summary_path) if task_summary_path else None,
        "all_jobs_fold_metrics_json": str(fold_metrics_path) if fold_metrics_path else None,
        "completed_jobs": completed_jobs,
    }
    summary_path = log_dir / f"queue_summary_{timestamp}.json"
    save_json(summary_path, summary)
    print(f"[QUEUE_DONE] summary={summary_path}")


if __name__ == "__main__":
    main()
