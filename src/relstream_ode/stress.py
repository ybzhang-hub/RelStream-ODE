"""Causally valid temporal stress tests; immutable availability and answer groups."""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .config import source_hashes
from .data import HistoryBuilder
from .evaluation import evaluate
from .trainer import add_data_arguments, load_evaluation
from .utils import dump_json, fingerprint, resolve_device, synchronize


def perturbation_plan(data, seed, std_steps, granularity="events"):
    """Common random numbers and nested affected units, without changing truth times."""
    rng = np.random.default_rng(seed)
    if granularity == "events":
        count = len(data.event_times["test"])
        return rng.permutation(count), rng.normal(0, std_steps, count)
    if granularity == "timestamps":
        count = len(data.split_times["test"])
        return rng.permutation(count), rng.normal(0, std_steps, count)
    raise ValueError("Invalid perturbation granularity")


def main():
    parser = argparse.ArgumentParser(
        description="Leakage-safe horizon evaluation and time-feature perturbations"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--experiment", choices=["horizons", "jitter"], required=True)
    parser.add_argument(
        "--horizon-mode", choices=["observed_targets", "fixed_label_shift"]
    )
    parser.add_argument(
        "--offset-unit", choices=["steps", "snapshots", "days"], default="steps"
    )
    parser.add_argument("--offsets", type=float, nargs="+", default=[10, 20, 30, 40, 50])
    parser.add_argument("--rates", type=float, nargs="+", default=[20, 30, 40, 50, 60])
    parser.add_argument("--noise-std-steps", type=float, default=3.0)
    parser.add_argument(
        "--noise-granularity", choices=["events", "timestamps"], default="events",
        help="'events' matches fact percentages, but can require one ODE pass per noisy query; "
             "'timestamps' is cheaper and measures a percentage of timestamp groups",
    )
    parser.add_argument("--noise-seed", type=int, default=2026)
    parser.add_argument("--smoke", action="store_true")
    add_data_arguments(parser)
    args = parser.parse_args()
    if args.experiment == "horizons" and args.horizon_mode is None:
        parser.error("--horizon-mode is required for horizon tests")
    if not np.isfinite(args.noise_std_steps) or args.noise_std_steps < 0:
        parser.error("--noise-std-steps must be finite and nonnegative")
    if any(not np.isfinite(x) or not 0 <= x <= 100 for x in args.rates):
        parser.error("--rates must be finite percentages between 0 and 100")
    if any(not np.isfinite(x) or x <= 0 for x in args.offsets):
        parser.error("--offsets must be finite and strictly positive")
    device = resolve_device(args.device)
    payload, digest, data, model = load_evaluation(
        args.checkpoint, args.data_root, args.cache_root, device, args.threads
    )
    if data.name != "GDELT":
        raise ValueError("Stress tests in this project target processed GDELT")
    if data.time_coordinate != "benchmark_steps" and args.offset_unit in {"steps", "snapshots"}:
        raise ValueError(
            "An old days-based checkpoint cannot use step-valued offsets without retraining. "
            "Use a new benchmark_steps checkpoint or explicitly request --offset-unit days."
        )
    train_raw_times = np.unique(data.raw["train"][:, 3])
    raw_step = float(np.gcd.reduce(np.diff(train_raw_times)))
    step_width = raw_step / data.units
    diagnostic = bool(args.smoke or payload["diagnostic"])
    limits = (
        {"timestamps": 2, "queries": 32, "graph_rows": 1024}
        if diagnostic else {"timestamps": 0, "queries": 0, "graph_rows": 0}
    )
    width = step_width if diagnostic else payload["config"]["history_steps"]
    noise_std_in_model_units = args.noise_std_steps * step_width
    setting = {
        "experiment": args.experiment,
        "horizon_mode": args.horizon_mode,
        "offset_unit": args.offset_unit,
        "offsets": args.offsets,
        "rates": args.rates,
        "noise_std_steps": args.noise_std_steps,
        "noise_std_in_model_units": noise_std_in_model_units,
        "noise_granularity": args.noise_granularity,
        "noise_seed": args.noise_seed,
        "history_window_in_model_units": width,
        "model_time_coordinate": data.time_coordinate,
        "diagnostic": diagnostic,
    }
    destination = (
        args.output_root
        / ("stress_diagnostic" if diagnostic else "stress")
        / payload["config"]["variant"]
        / data.name
        / f"seed_{payload['seed']}"
        / (digest[:12] + "_" + fingerprint(setting)[:12])
    )
    if (destination / "result.json").exists():
        raise FileExistsError("This stress run already exists")
    destination.mkdir(parents=True, exist_ok=True)
    rows = []
    if args.experiment == "horizons":
        # Conversion relative to the chosen model-time axis. E.g. one GDELT
        # step = 15 raw minutes, so 1 day = 96 benchmark steps.
        offsets = [
            x * (1.0 if args.offset_unit in {"steps", "snapshots"}
                 else data.raw_units_per_day / data.units)
            for x in args.offsets
        ]
        times = list(data.split_times["test"])
        if args.horizon_mode == "observed_targets":
            times = [t for t in times if t - max(offsets) >= data.observation_times[0]]
        if not times:
            rows = [
                {
                    "offset": value, "offset_in_model_units": delta,
                    "status": "not_evaluable",
                    "reason": "No target cohort has observed history at every lead time",
                }
                for value, delta in zip(args.offsets, offsets)
            ]
        else:
            for value, delta in zip(args.offsets, offsets):
                builder = HistoryBuilder(data, device, width, limits["graph_rows"])
                synchronize(device)
                tick = time.perf_counter()
                result = evaluate(
                    model, data, builder, "test", payload["config"]["batch_size"],
                    payload["protocol"]["tie_handling"],
                    limits["timestamps"], limits["queries"],
                    offset_steps=delta if args.horizon_mode == "observed_targets" else None,
                    shift_query_steps=delta if args.horizon_mode == "fixed_label_shift" else None,
                    history_steps_override=width, selected_times=times,
                )
                synchronize(device)
                rows.append({
                    "offset": value, "offset_in_model_units": delta,
                    "status": "completed", "seconds": time.perf_counter()-tick,
                    **result,
                })
                dump_json(destination / "progress.json", rows)
    else:
        ordering, noise = perturbation_plan(
            data, args.noise_seed, noise_std_in_model_units, args.noise_granularity
        )
        for rate in args.rates:
            selected = ordering[:int(len(ordering) * rate / 100)]
            offsets = np.zeros(len(data.event_times["test"]), dtype=np.float64)
            if args.noise_granularity == "events":
                offsets[selected] = noise[selected]
            else:
                timestamps = np.asarray(list(data.split_times["test"]), dtype=np.float64)
                for i in selected:
                    offsets[data.event_times["test"] == timestamps[i]] = noise[i]
            # 'perturbed' stores query-only time-code offsets. The event index,
            # availability times, label times and filtered answers remain original.
            perturbed = data.perturbed(data.event_times["test"] + offsets)
            builder = HistoryBuilder(perturbed, device, width, limits["graph_rows"])
            synchronize(device)
            tick = time.perf_counter()
            result = evaluate(
                model, perturbed, builder, "test", payload["config"]["batch_size"],
                payload["protocol"]["tie_handling"], limits["timestamps"],
                limits["queries"],
            )
            synchronize(device)
            np.savez_compressed(
                destination / f"jitter_{rate:g}.npz",
                original_target_times=data.event_times["test"],
                model_time_code_offsets=offsets,
                selected_unit_ids=selected,
            )
            rows.append({
                "rate_percent": rate,
                "perturbed_units": int(len(selected)),
                "granularity": args.noise_granularity,
                "status": "completed", "seconds": time.perf_counter()-tick,
                **result,
            })
            dump_json(destination / "progress.json", rows)
    result = {
        "status": "completed_stress_evaluation",
        "dataset": data.name,
        "variant": payload["config"]["variant"],
        "seed": payload["seed"],
        "diagnostic": diagnostic,
        "checkpoint_sha256": digest,
        "data_fingerprint": data.fingerprint,
        "model_config_fingerprint": fingerprint(payload["config"]),
        "protocol_fingerprint": fingerprint(payload["protocol"]),
        "source_sha256": source_hashes(),
        "settings": setting,
        "rows": rows,
        "target_labels": (
            "original_observed_target_times" if args.horizon_mode == "observed_targets"
            else "original_event_labels_at_synthetic_shifted_query_times"
            if args.experiment == "horizons"
            else "original_event_labels_and_filter_groups_unchanged"
        ),
        "noise_policy": (
            "query time-code features shifted; event availability, graph histories, "
            "integration boundaries and true-label filtering frozen at original times; "
            "nested affected groups and shared noise across seeds"
        ),
    }
    dump_json(destination / "result.json", result)
    print(json.dumps(result), flush=True)
