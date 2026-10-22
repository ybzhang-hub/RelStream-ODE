"""Full-entity optimization, validation selection and resumable training."""

import argparse
try:
    import fcntl
except ImportError:  # Native Windows; file locking handled below.
    fcntl = None
import json
import signal
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .config import (
    DATA_ROOT,
    DATASETS,
    ROOT,
    SEEDS,
    VARIANTS,
    load_config,
    checkpoint_config,
    source_hashes,
    validate_config,
)
from .data import HistoryBuilder, TemporalDataset
from .evaluation import evaluate
from .model import RelStreamODE
from .utils import (
    configure_seed,
    dump_json,
    fingerprint,
    hardware,
    memory_snapshot,
    random_state,
    read_checkpoint,
    resolve_device,
    restore_random_state,
    save_checkpoint,
    synchronize,
)


def protocol(tie="average", config=None):
    config = config or load_config()
    return {
        "query_direction": "tail_only",
        "inverse_edges": False,
        "candidate_entities": "full_released_vocabulary",
        "filter_key": ["subject", "relation", "target_time"],
        "filter_truth": "all_true_tails_at_the_target_time; target_retained",
        "tie_handling": tie,
        "objective": "full_entity_softmax_cross_entropy_mean; no_negative_sampling",
        "history": "rolling_observed_events_strictly_before_target",
        "initialization": "cumulative_t_le_first_boundary",
        "intervals": "new_events_since_previous_boundary",
        "time_coordinate": config["time_coordinate"],
        "history_width": config["history_steps"],
        "transport_formula": config["transport_formula"],
        "transport_implementation": config["transport_implementation"],
        "usage_aggregation": config["usage_aggregation"],
        "rk4_substeps": config["rk4_substeps"],
        "solver": "classical_RK4; direct_differentiation",
        "checkpoint": "maximum_validation_MRR; earliest_checkpoint_retained_for_equal_MRR",
        "early_stopping": False,
    }


@contextmanager
def locked(directory):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".run.lock").open("a") as stream:
        try:
            if fcntl is not None:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            else:
                import msvcrt
                stream.write(" ")
                stream.flush()
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        except (BlockingIOError, OSError):
            raise RuntimeError(
                "This run directory already has an active writer"
            ) from None
        try:
            yield
        finally:
            if fcntl is None:
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)


def add_data_arguments(parser):
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--cache-root", type=Path, default=ROOT / "cache")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs")


def train_arguments():
    parser = argparse.ArgumentParser(
        description="Train RelStream-ODE with validation-only checkpoint selection"
    )
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, default=42)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/full.json")
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Bounded execution check; excluded from full-run summaries",
    )
    parser.add_argument(
        "--validation-only",
        action="store_true",
        help="Configuration selection without test evaluation",
    )
    parser.add_argument(
        "--tie", choices=["average", "optimistic", "pessimistic"], default="average"
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=10,
        help="Save full recovery state after this many completed timestamps",
    )
    parser.add_argument(
        "--checkpoint-segments",
        type=int,
        default=8,
        help="Intervals per activation-checkpoint segment; 0 disables recomputation",
    )
    add_data_arguments(parser)
    return parser.parse_args()


def _limits(data, smoke):
    return (
        {"timestamps": 2, "queries_per_timestamp": 32, "graph_rows": 1024}
        if smoke
        else {"timestamps": 0, "queries_per_timestamp": 0, "graph_rows": 0}
    )


def _save_ranks(path, records):
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **records)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def train(args):
    if args.threads < 1 or args.checkpoint_every < 1 or args.checkpoint_segments < 0:
        raise ValueError(
            "Invalid thread, checkpoint interval or checkpoint segment setting"
        )
    device = resolve_device(args.device)
    configure_seed(args.seed, args.threads)
    config = load_config(args.config)
    if args.variant is not None:
        config["variant"] = args.variant
        if args.variant == "max_usage":
            config["usage_aggregation"] = "max"
        elif args.variant == "attention_usage":
            config["usage_aggregation"] = "attention"
    data_tick = time.perf_counter()
    data = TemporalDataset(args.dataset, args.data_root, args.cache_root, config["time_coordinate"])
    limits = _limits(data, args.smoke)
    if args.smoke:
        config.update(
            epochs=1, history_steps=float(np.diff(data.observation_times).min())
        )
    validate_config(config)
    builder = HistoryBuilder(data, device, config["history_steps"], limits["graph_rows"])
    p = protocol(args.tie, config)
    runtime = {
        "device_type": device.type,
        "threads": args.threads,
        "checkpoint_segments": args.checkpoint_segments,
        "limits": limits,
        "validation_only": args.validation_only,
    }
    destination = (
        args.output_root
        / ("diagnostic_runs" if args.smoke else "runs")
        / config["variant"]
        / args.dataset
        / f"seed_{args.seed}"
    )
    with locked(destination):
        if (destination / "result.json").exists():
            raise FileExistsError(
                "Completed run exists; evaluate its checkpoint or select another output root"
            )
        if (destination / "last.pt").exists() and not args.resume:
            raise FileExistsError("An interrupted run exists; use --resume")
        if args.resume and not (destination / "last.pt").exists():
            raise FileNotFoundError("No recovery checkpoint exists")
        model = RelStreamODE(data.entities, data.relations, config).to(device)
        optimizer = torch.optim.Adam(
            [x for x in model.parameters() if x.requires_grad],
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"],
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config["scheduler_step"],
            gamma=config["scheduler_gamma"],
        )
        provenance = {
            "dataset": args.dataset,
            "seed": args.seed,
            "config": config,
            "protocol": p,
            "data_fingerprint": data.fingerprint,
            "source_sha256": source_hashes(),
            "runtime": runtime,
        }
        (
            best,
            best_epoch,
            epoch_rows,
            next_epoch,
            progress,
            wall_before,
            optimizer_steps,
        ) = None, 0, [], 1, None, 0.0, 0
        if args.resume:
            state, _ = read_checkpoint(destination / "last.pt")
            for key, value in provenance.items():
                if state[key] != value:
                    raise ValueError(f"Recovery provenance mismatch: {key}")
            model.load_state_dict(state["state_dict"])
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            (
                best,
                best_epoch,
                epoch_rows,
                next_epoch,
                progress,
                wall_before,
                optimizer_steps,
            ) = (
                state[k]
                for k in [
                    "best_validation_MRR",
                    "best_epoch",
                    "epoch_rows",
                    "next_epoch",
                    "progress",
                    "training_wall_seconds",
                    "optimizer_steps",
                ]
            )
            restore_random_state(state["random_state"])
        synchronize(device)
        info = {
            "dataset": args.dataset,
            "seed": args.seed,
            "variant": config["variant"],
            "status": "running",
            "diagnostic": args.smoke,
            "full_training": not args.smoke,
            "full_test": False,
            "validation_only": args.validation_only,
            "config": "config.json",
            "checkpoint": "best.pt",
            "data_fingerprint": data.fingerprint,
            "protocol_fingerprint": fingerprint(p),
            "source_sha256": provenance["source_sha256"],
            "hardware": hardware(device),
            "preprocessing_seconds": time.perf_counter() - data_tick,
            "trainable_parameters": sum(
                x.numel() for x in model.parameters() if x.requires_grad
            ),
            "total_parameters": sum(x.numel() for x in model.parameters()),
            "limits": limits,
        }
        for filename, value in [
            ("config.json", config),
            ("protocol.json", p),
            ("status.json", info),
        ]:
            dump_json(destination / filename, value)
        wall_tick = time.perf_counter()
        stop = {"requested": False}

        def request_stop(signum, frame):
            stop["requested"] = True

        old_handlers = {
            sig: signal.signal(sig, request_stop)
            for sig in [signal.SIGINT, signal.SIGTERM]
        }

        def save_recovery(epoch, current):
            save_checkpoint(
                destination / "last.pt",
                {
                    **provenance,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "random_state": random_state(),
                    "best_validation_MRR": best,
                    "best_epoch": best_epoch,
                    "epoch_rows": epoch_rows,
                    "next_epoch": epoch,
                    "progress": current,
                    "training_wall_seconds": wall_before
                    + time.perf_counter()
                    - wall_tick,
                    "optimizer_steps": optimizer_steps,
                },
            )

        train_times = [
            t for t in data.split_times["train"] if t > data.observation_times[0]
        ][: limits["timestamps"] or None]
        info["training_warmup_timestamp"] = float(data.observation_times[0])
        try:
            for epoch in range(next_epoch, config["epochs"] + 1):
                model.train()
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                if progress is None:
                    times = [
                        train_times[i] for i in np.random.permutation(len(train_times))
                    ]
                    progress = {
                        "times": times,
                        "next_time": 0,
                        "loss_sum": 0.0,
                        "queries": 0,
                        "steps": 0,
                        "gradient_norm_sum": 0.0,
                        "above_clip": 0,
                        "integrated_intervals": 0,
                        "vector_field_evaluations": 0,
                        "training_seconds": 0.0,
                        "peak_allocated_bytes": 0,
                        "peak_reserved_bytes": 0,
                    }
                times = progress["times"]
                lr = optimizer.param_groups[0]["lr"]
                synchronize(device)
                tick = time.perf_counter()
                for i in range(progress["next_time"], len(times)):
                    target = times[i]
                    rows = data.split_times["train"][target][
                        : limits["queries_per_timestamp"] or None
                    ]
                    order = np.random.permutation(len(rows))
                    history = builder.build(target)
                    for begin in range(0, len(rows), config["batch_size"]):
                        batch = torch.as_tensor(
                            rows[order[begin : begin + config["batch_size"]]].copy(),
                            device=device,
                        )
                        optimizer.zero_grad(set_to_none=True)
                        nfe_before = model.field.nfe
                        scores = model(history, batch, args.checkpoint_segments)
                        loss = F.cross_entropy(scores, batch[:, 2])
                        if not torch.isfinite(loss):
                            raise FloatingPointError(
                                f"Nonfinite loss at epoch {epoch}, day {target}"
                            )
                        loss.backward()
                        progress["integrated_intervals"] += len(history.intervals)
                        progress["vector_field_evaluations"] += model.field.nfe - nfe_before
                        norm = torch.nn.utils.clip_grad_norm_(
                            model.parameters(),
                            config["gradient_clip"],
                            error_if_nonfinite=True,
                        )
                        optimizer.step()
                        optimizer_steps += 1
                        progress["loss_sum"] += float(loss.detach()) * len(batch)
                        progress["queries"] += len(batch)
                        progress["steps"] += 1
                        progress["gradient_norm_sum"] += float(norm)
                        progress["above_clip"] += int(norm > config["gradient_clip"])
                    progress["next_time"] = i + 1
                    if (i + 1) % args.checkpoint_every == 0 or stop["requested"]:
                        synchronize(device)
                        progress["training_seconds"] += time.perf_counter() - tick
                        for key, value in memory_snapshot(device).items():
                            progress[key] = max(progress[key], value)
                        save_recovery(epoch, progress)
                        info.update(
                            epoch=epoch,
                            completed_timestamps=i + 1,
                            total_timestamps=len(times),
                            best_epoch=best_epoch,
                            validation_MRR=best,
                        )
                        dump_json(destination / "status.json", info)
                        print(
                            json.dumps(
                                {
                                    "epoch": epoch,
                                    "completed_timestamps": i + 1,
                                    "total_timestamps": len(times),
                                    "cross_entropy": progress["loss_sum"]
                                    / max(progress["queries"], 1),
                                }
                            ),
                            flush=True,
                        )
                        if stop["requested"]:
                            raise KeyboardInterrupt(
                                "Recovery checkpoint saved at a timestamp boundary"
                            )
                        synchronize(device)
                        tick = time.perf_counter()
                synchronize(device)
                progress["training_seconds"] += time.perf_counter() - tick
                for key, value in memory_snapshot(device).items():
                    progress[key] = max(progress[key], value)
                synchronize(device)
                validation_tick = time.perf_counter()
                validation = evaluate(
                    model,
                    data,
                    builder,
                    "valid",
                    config["batch_size"],
                    args.tie,
                    limits["timestamps"],
                    limits["queries_per_timestamp"],
                )
                synchronize(device)
                validation_seconds = time.perf_counter() - validation_tick
                if best is None or validation["MRR"] > best:
                    best, best_epoch = validation["MRR"], epoch
                    save_checkpoint(
                        destination / "best.pt",
                        {
                            **provenance,
                            "state_dict": model.state_dict(),
                            "epoch": epoch,
                            "validation_MRR": best,
                            "diagnostic": args.smoke,
                            "limits": limits,
                        },
                    )
                record = {
                    "epoch": epoch,
                    "cross_entropy": progress["loss_sum"] / progress["queries"],
                    "training_queries": progress["queries"],
                    "optimizer_steps": progress["steps"],
                    "optimizer_steps_total": optimizer_steps,
                    "integrated_intervals": progress["integrated_intervals"],
                    "vector_field_evaluations": progress["vector_field_evaluations"],
                    "learning_rate": lr,
                    "pre_clip_gradient_norm_mean": progress["gradient_norm_sum"]
                    / progress["steps"],
                    "gradient_norm_above_clip_fraction": progress["above_clip"]
                    / progress["steps"],
                    "training_seconds": progress["training_seconds"],
                    "validation_seconds": validation_seconds,
                    "validation_MRR": validation["MRR"],
                    "best_validation_MRR": best,
                    "best_epoch": best_epoch,
                    **{
                        key: progress[key]
                        for key in ["peak_allocated_bytes", "peak_reserved_bytes"]
                    },
                }
                epoch_rows.append(record)
                scheduler.step()
                progress = None
                dump_json(destination / "epochs.json", epoch_rows)
                save_recovery(epoch + 1, None)
                print(json.dumps(record), flush=True)
            info["training_wall_seconds"] = (
                wall_before + time.perf_counter() - wall_tick
            )
            info.update(
                best_epoch=best_epoch,
                validation_MRR=best,
                epochs_completed=len(epoch_rows),
            )
            chosen, chosen_hash = read_checkpoint(destination / "best.pt")
            model.load_state_dict(chosen["state_dict"])
            del chosen
            info["checkpoint_sha256"] = chosen_hash
            optimizer = None
            scheduler = None
            if not args.validation_only:
                collector = {}
                synchronize(device)
                tick = time.perf_counter()
                result = evaluate(
                    model,
                    data,
                    builder,
                    "test",
                    config["batch_size"],
                    args.tie,
                    limits["timestamps"],
                    limits["queries_per_timestamp"],
                    collector,
                )
                synchronize(device)
                info.update(
                    result,
                    test_seconds=time.perf_counter() - tick,
                    full_test=not args.smoke,
                )
                _save_ranks(destination / "test_ranks.npz", collector)
            info["status"] = (
                "completed_diagnostic"
                if args.smoke
                else "completed_validation"
                if args.validation_only
                else "completed"
            )
            dump_json(destination / "result.json", info)
            dump_json(destination / "status.json", info)
            print(json.dumps(info), flush=True)
        except BaseException as error:
            info.update(
                status="interrupted"
                if isinstance(error, KeyboardInterrupt)
                else "failed",
                error=f"{type(error).__name__}: {error}",
            )
            dump_json(destination / "status.json", info)
            raise
        finally:
            for sig, handler in old_handlers.items():
                signal.signal(sig, handler)


def main():
    train(train_arguments())


def load_evaluation(checkpoint, data_root, cache_root, device, threads):
    payload, digest = read_checkpoint(checkpoint)
    payload["config"] = checkpoint_config(payload["config"])
    configure_seed(payload["seed"], threads)
    data = TemporalDataset(
        payload["dataset"], data_root, cache_root, payload["config"]["time_coordinate"]
    )
    if payload["data_fingerprint"] != data.fingerprint:
        raise ValueError("Checkpoint data fingerprint mismatch")
    model = RelStreamODE(data.entities, data.relations, payload["config"]).to(device)
    model.load_state_dict(payload["state_dict"])
    return payload, digest, data, model


def evaluate_main():
    parser = argparse.ArgumentParser(
        description="Independently evaluate a validation-selected checkpoint"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=["valid", "test"], default="test")
    parser.add_argument(
        "--full-split",
        action="store_true",
        help="Evaluate all timestamps/queries, including for a diagnostic checkpoint",
    )
    add_data_arguments(parser)
    args = parser.parse_args()
    if args.threads < 1:
        raise ValueError("threads must be positive")
    device = resolve_device(args.device)
    payload, digest, data, model = load_evaluation(
        args.checkpoint, args.data_root, args.cache_root, device, args.threads
    )
    limits = payload["limits"]
    builder = HistoryBuilder(
        data, device, payload["config"]["history_steps"], limits["graph_rows"]
    )
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = (
        args.output_root
        / "evaluations"
        / payload["config"]["variant"]
        / data.name
        / f"seed_{payload['seed']}"
        / (digest[:12] + "_" + timestamp)
    )
    collector = {}
    synchronize(device)
    tick = time.perf_counter()
    result = evaluate(
        model,
        data,
        builder,
        args.split,
        payload["config"]["batch_size"],
        payload["protocol"]["tie_handling"],
        0 if args.full_split else limits["timestamps"],
        0 if args.full_split else limits["queries_per_timestamp"],
        collector,
    )
    synchronize(device)
    info = {
        "status": "completed_evaluation",
        "dataset": data.name,
        "seed": payload["seed"],
        "variant": payload["config"]["variant"],
        "split": args.split,
        "diagnostic": payload["diagnostic"],
        "full_test": args.split == "test"
        and (args.full_split or not payload["diagnostic"]),
        "best_epoch": payload["epoch"],
        "validation_MRR": payload["validation_MRR"],
        "checkpoint_sha256": digest,
        "data_fingerprint": data.fingerprint,
        "source_sha256": source_hashes(),
        "training_source_sha256": payload["source_sha256"],
        "seconds": time.perf_counter() - tick,
        "hardware": hardware(device),
        **result,
    }
    dump_json(destination / "config.json", payload["config"])
    dump_json(destination / "protocol.json", payload["protocol"])
    dump_json(destination / "result.json", info)
    _save_ranks(destination / "ranks.npz", collector)
    print(json.dumps(info), flush=True)
