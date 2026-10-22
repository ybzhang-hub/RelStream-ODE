"""Bounded parallel experiment queues with provenance-aware recovery."""

import argparse
import itertools
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from .config import (
    DATA_ROOT,
    DATASETS,
    ROOT,
    SEEDS,
    VARIANTS,
    CORE_VARIANTS,
    load_config,
    source_hashes,
)
from .trainer import protocol
from .utils import dump_json, fingerprint, sha256


def compatible_result(
    directory, dataset, seed, config, validation_only=False, smoke=False
):
    result = json.loads((directory / "result.json").read_text())
    saved = json.loads((directory / "config.json").read_text())
    expected_status = (
        "completed_diagnostic"
        if smoke
        else "completed_validation"
        if validation_only
        else "completed"
    )
    reference = json.loads((ROOT / "splits" / f"{dataset}.json").read_text())
    if (
        result["status"] != expected_status
        or result["dataset"] != dataset
        or result["seed"] != seed
        or saved != config
        or result["protocol_fingerprint"] != fingerprint(protocol(config=config))
        or result["data_fingerprint"] != fingerprint(reference)
        or result["source_sha256"] != source_hashes()
        or result["checkpoint_sha256"] != sha256(directory / "best.pt")
    ):
        raise ValueError(
            f"Completed run is incompatible with this request: {directory}"
        )
    return result


def stop_children(active):
    for process, stream in active.values():
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    for process, stream in active.values():
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
        stream.close()


def main():
    parser = argparse.ArgumentParser(
        description="Run repeated seeds and component ablations"
    )
    parser.add_argument(
        "--datasets", nargs="+", choices=DATASETS, default=list(DATASETS)
    )
    parser.add_argument(
        "--variants", nargs="+", choices=VARIANTS, default=list(CORE_VARIANTS)
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, choices=SEEDS, default=list(SEEDS)
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--cache-root", type=Path, default=ROOT / "cache")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs")
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    for values in [args.datasets, args.variants, args.seeds]:
        if len(values) != len(set(values)):
            raise ValueError("Duplicate jobs are not allowed")
    for name in ["data_root", "cache_root", "output_root"]:
        setattr(args, name, getattr(args, name).resolve())
    if args.config is not None:
        args.config = args.config.resolve()
    pending = list(itertools.product(args.variants, args.datasets, args.seeds))
    active, records = {}, {}
    logs = args.output_root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    environment = {
        **os.environ,
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "OMP_NUM_THREADS": "4",
        "PYTHONUNBUFFERED": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    }
    try:
        while pending or active:
            for job, (process, stream) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                stream.close()
                del active[job]
                if code:
                    records[job].update(status="failed", exit_code=code)
                    raise RuntimeError(
                        f"Run failed: {job}; inspect its log and resume the queue"
                    )
                variant, dataset, seed = job
                folder = args.output_root / "runs" / variant / dataset / f"seed_{seed}"
                cfg = load_config(args.config or ROOT / "configs" / f"{variant}.json")
                cfg["variant"] = variant
                compatible_result(folder, dataset, seed, cfg, args.validation_only)
                records[job]["status"] = "completed"
            while pending and len(active) < args.workers:
                job = pending.pop(0)
                variant, dataset, seed = job
                config_path = args.config or ROOT / "configs" / f"{variant}.json"
                cfg = load_config(config_path)
                cfg["variant"] = variant
                folder = args.output_root / "runs" / variant / dataset / f"seed_{seed}"
                if (folder / "result.json").exists():
                    compatible_result(folder, dataset, seed, cfg, args.validation_only)
                    records[job] = {
                        "variant": variant,
                        "dataset": dataset,
                        "seed": seed,
                        "status": "completed",
                    }
                    continue
                command = [
                    sys.executable,
                    str(ROOT / "train.py"),
                    "--dataset",
                    dataset,
                    "--variant",
                    variant,
                    "--seed",
                    str(seed),
                    "--config",
                    str(config_path),
                    "--device",
                    args.device,
                    "--data-root",
                    str(args.data_root),
                    "--cache-root",
                    str(args.cache_root),
                    "--output-root",
                    str(args.output_root),
                ]
                if (folder / "last.pt").exists():
                    command.append("--resume")
                if args.validation_only:
                    command.append("--validation-only")
                stream = (logs / f"{variant}_{dataset}_{seed}.log").open("a")
                process = subprocess.Popen(
                    command,
                    cwd=ROOT,
                    env=environment,
                    stdin=subprocess.DEVNULL,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                active[job] = (process, stream)
                records[job] = {
                    "variant": variant,
                    "dataset": dataset,
                    "seed": seed,
                    "status": "running",
                    "pid": process.pid,
                }
            dump_json(
                logs / "queue.json",
                {
                    "status": "running",
                    "active": len(active),
                    "pending": len(pending),
                    "workers": args.workers,
                    "jobs": list(records.values()),
                },
            )
            if active:
                time.sleep(2)
        dump_json(
            logs / "queue.json",
            {
                "status": "completed",
                "active": 0,
                "pending": 0,
                "jobs": list(records.values()),
            },
        )
    except BaseException as error:
        dump_json(
            logs / "queue.json",
            {
                "status": "interrupted"
                if isinstance(error, KeyboardInterrupt)
                else "failed",
                "error": str(error),
                "jobs": list(records.values()),
            },
        )
        raise
    finally:
        stop_children(active)
