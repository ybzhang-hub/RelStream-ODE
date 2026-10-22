"""Synchronized serial resource measurements with explicit phase boundaries."""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .data import HistoryBuilder
from .evaluation import evaluate
from .trainer import add_data_arguments, load_evaluation
from .utils import (
    dump_json,
    hardware,
    memory_snapshot,
    random_state,
    resolve_device,
    restore_random_state,
    synchronize,
)


def _exclusive_gpu(device):
    if device.type != "cuda":
        return False
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
    )
    others = {
        int(x.strip()) for x in result.stdout.splitlines() if x.strip().isdigit()
    } - {os.getpid()}
    if others:
        raise RuntimeError(f"Other GPU processes are active: {sorted(others)}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Serial training/validation/test resource measurements"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--checkpoint-segments", type=int, default=8)
    add_data_arguments(parser)
    args = parser.parse_args()
    if args.repeats < 1 or args.checkpoint_segments < 0:
        raise ValueError("Invalid repetition or checkpoint setting")
    device = resolve_device(args.device)
    exclusive = _exclusive_gpu(device)
    payload, digest, data, model = load_evaluation(
        args.checkpoint, args.data_root, args.cache_root, device, args.threads
    )
    directory = (
        args.output_root
        / ("resources_diagnostic" if payload["diagnostic"] else "resources")
        / payload["config"]["variant"]
        / data.name
        / f"seed_{payload['seed']}"
    )
    if (directory / "result.json").exists():
        raise FileExistsError("A resource record exists; choose another output root")
    limits = payload["limits"]
    config = payload["config"]
    builder = HistoryBuilder(data, device, config["history_steps"], limits["graph_rows"])
    dates = [t for t in data.split_times["train"] if t > data.observation_times[0]][
        : limits["timestamps"] or None
    ]
    selected = {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }
    rng = random_state()
    model.train()
    first = dates[0]
    batch = torch.as_tensor(
        data.split_times["train"][first][: config["batch_size"]].copy(), device=device
    )
    loss = F.cross_entropy(
        model(builder.build(first), batch, args.checkpoint_segments), batch[:, 2]
    )
    loss.backward()
    model.zero_grad(set_to_none=True)
    model.load_state_dict(selected)
    restore_random_state(rng)
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    synchronize(device)
    tick = time.perf_counter()
    training_queries = 0
    for target in dates:
        rows = data.split_times["train"][target][
            : limits["queries_per_timestamp"] or None
        ]
        order = np.random.permutation(len(rows))
        history = builder.build(target)
        for begin in range(0, len(rows), config["batch_size"]):
            batch = torch.as_tensor(
                rows[order[begin : begin + config["batch_size"]]].copy(), device=device
            )
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(
                model(history, batch, args.checkpoint_segments), batch[:, 2]
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config["gradient_clip"], error_if_nonfinite=True
            )
            optimizer.step()
            training_queries += len(batch)
    synchronize(device)
    train_seconds = time.perf_counter() - tick
    training_peak = memory_snapshot(device)
    optimizer = None
    model.zero_grad(set_to_none=True)
    model.load_state_dict(selected)
    del selected
    if device.type == "cuda":
        torch.cuda.empty_cache()
    model.eval()
    phase_values = {"valid": [], "test": []}
    inference_peaks = []
    for split in ["valid", "test"]:
        evaluate(
            model,
            data,
            builder,
            split,
            config["batch_size"],
            payload["protocol"]["tie_handling"],
            min(3, limits["timestamps"]) if limits["timestamps"] else 3,
            limits["queries_per_timestamp"],
        )
        for repeat in range(args.repeats):
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            synchronize(device)
            tick = time.perf_counter()
            result = evaluate(
                model,
                data,
                builder,
                split,
                config["batch_size"],
                payload["protocol"]["tie_handling"],
                limits["timestamps"],
                limits["queries_per_timestamp"],
            )
            synchronize(device)
            seconds = time.perf_counter() - tick
            phase_values[split].append(seconds)
            if split == "test":
                inference_peaks.append(memory_snapshot(device))
    record = {
        "status": "completed_resource_measurement",
        "dataset": data.name,
        "variant": config["variant"],
        "seed": payload["seed"],
        "diagnostic": payload["diagnostic"],
        "checkpoint_sha256": digest,
        "data_fingerprint": data.fingerprint,
        "model_config": config,
        "activation_checkpoint_intervals": args.checkpoint_segments,
        "hardware": hardware(device),
        "exclusive_gpu": exclusive,
        "trainable_parameters": sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ),
        "training_seconds_per_epoch": [train_seconds],
        "validation_seconds": phase_values["valid"],
        "full_test_seconds": phase_values["test"],
        "inference_ms_per_query": [
            1000 * x / result["queries"] for x in phase_values["test"]
        ],
        "training_queries": training_queries,
        "test_queries": result["queries"],
        "peak_allocated_bytes": [training_peak["peak_allocated_bytes"]],
        "peak_reserved_bytes": [training_peak["peak_reserved_bytes"]],
        "inference_peaks": inference_peaks,
        "scope": {
            "training": "one representative epoch from selected weights with fresh Adam; graph construction, transfers, encoding, CE, backward, clipping, update included",
            "validation_and_test": "complete specified split; causal history, encoding, full scoring and filtering included",
            "excludes": "initial data preparation, checkpoint loading, warm-up, checkpoint/rank-file I/O",
            "synchronization": "CUDA synchronize before and after each phase",
            "warmup": "training forward/backward then restore weights/RNG; three timestamps before each inference phase",
            "memory": "PyTorch allocator peak allocated/reserved bytes; includes resident weights/caches; excludes CUDA context and external allocations",
            "gradient_implementation": "direct differentiation; optional deterministic activation recomputation; no adjoint",
        },
    }
    dump_json(directory / "result.json", record)
    print(json.dumps(record), flush=True)
