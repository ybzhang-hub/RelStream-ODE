"""Explicit open-loop lead times, fixed-label shifts and timestamp noise."""

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


def perturbation_plan(data, seed, std_days):
    rng = np.random.default_rng(seed)
    count = len(data.event_times["test"])
    return rng.permutation(count), rng.normal(0, std_days, count)


def main():
    parser = argparse.ArgumentParser(
        description="Temporal stress evaluation with explicit time and target-label conventions"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--experiment", choices=["horizons", "jitter"], required=True)
    parser.add_argument(
        "--horizon-mode", choices=["observed_targets", "fixed_label_shift"]
    )
    parser.add_argument("--offset-unit", choices=["days", "snapshots"], default="days")
    parser.add_argument(
        "--offsets", type=float, nargs="+", default=[10, 20, 30, 40, 50]
    )
    parser.add_argument("--rates", type=float, nargs="+", default=[20, 30, 40, 50, 60])
    parser.add_argument("--noise-std-days", type=float, default=3.0)
    parser.add_argument("--noise-seed", type=int, default=2026)
    parser.add_argument("--smoke", action="store_true")
    add_data_arguments(parser)
    args = parser.parse_args()
    if args.experiment == "horizons" and args.horizon_mode is None:
        parser.error(
            "--horizon-mode is required: select observed target times or explicitly shifted labels"
        )
    if args.noise_std_days < 0 or not np.isfinite(args.noise_std_days):
        raise ValueError("Invalid noise scale")
    if any(not 0 <= r <= 100 for r in args.rates):
        raise ValueError("Noise rates must be percentages")
    if any(not np.isfinite(x) or x <= 0 for x in args.offsets):
        raise ValueError("Offsets must be finite and positive")
    device = resolve_device(args.device)
    payload, digest, data, model = load_evaluation(
        args.checkpoint, args.data_root, args.cache_root, device, args.threads
    )
    if data.name != "GDELT":
        raise ValueError("Temporal stress configurations use the GDELT benchmark")
    raw_times = np.unique(data.raw["train"][:, 3])
    bin_days = float(np.gcd.reduce(np.diff(raw_times)) / data.units)
    diagnostic = bool(args.smoke or payload["diagnostic"])
    limits = (
        {"timestamps": 2, "queries": 32, "graph_rows": 1024}
        if diagnostic
        else {"timestamps": 0, "queries": 0, "graph_rows": 0}
    )
    width = bin_days if diagnostic else 100.0
    setting = {
        "experiment": args.experiment,
        "horizon_mode": args.horizon_mode,
        "offset_unit": args.offset_unit,
        "offsets": args.offsets,
        "rates": args.rates,
        "noise_std_days": args.noise_std_days,
        "noise_seed": args.noise_seed,
        "history_days": width,
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
        raise FileExistsError(
            "This stress evaluation exists; choose another output root"
        )
    destination.mkdir(parents=True, exist_ok=True)
    rows = []
    if args.experiment == "horizons":
        days = [
            x * (bin_days if args.offset_unit == "snapshots" else 1.0)
            for x in args.offsets
        ]
        times = list(data.split_times["test"])
        if args.horizon_mode == "observed_targets":
            times = [t for t in times if t - max(days) >= data.observation_times[0]]
        if not times:
            rows = [
                {
                    "offset": value,
                    "offset_days": delta,
                    "status": "not_evaluable",
                    "reason": "No matched target cohort has observed history at every requested lead time",
                }
                for value, delta in zip(args.offsets, days)
            ]
        else:
            for value, delta in zip(args.offsets, days):
                builder = HistoryBuilder(data, device, width, limits["graph_rows"])
                synchronize(device)
                tick = time.perf_counter()
                result = evaluate(
                    model,
                    data,
                    builder,
                    "test",
                    payload["config"]["batch_size"],
                    payload["protocol"]["tie_handling"],
                    limits["timestamps"],
                    limits["queries"],
                    offset_days=delta
                    if args.horizon_mode == "observed_targets"
                    else None,
                    shift_query_days=delta
                    if args.horizon_mode == "fixed_label_shift"
                    else None,
                    history_days_override=width,
                    selected_times=times,
                )
                synchronize(device)
                rows.append(
                    {
                        "offset": value,
                        "offset_days": delta,
                        "status": "completed",
                        "seconds": time.perf_counter() - tick,
                        **result,
                    }
                )
                dump_json(destination / "progress.json", rows)
    else:
        ordering, noise = perturbation_plan(data, args.noise_seed, args.noise_std_days)
        for rate in args.rates:
            selected = ordering[: int(len(ordering) * rate / 100)]
            times = data.event_times["test"].copy()
            times[selected] += noise[selected]
            perturbed = data.perturbed(times)
            builder = HistoryBuilder(perturbed, device, width, limits["graph_rows"])
            synchronize(device)
            tick = time.perf_counter()
            result = evaluate(
                model,
                perturbed,
                builder,
                "test",
                payload["config"]["batch_size"],
                payload["protocol"]["tie_handling"],
                limits["timestamps"],
                limits["queries"],
            )
            synchronize(device)
            np.savez_compressed(
                destination / f"jitter_{rate:g}.npz",
                test_days=times,
                changed_row_ids=selected,
            )
            rows.append(
                {
                    "rate_percent": rate,
                    "status": "completed",
                    "perturbed_facts": len(selected),
                    "seconds": time.perf_counter() - tick,
                    **result,
                }
            )
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
        "target_labels": "original_observed_target_times"
        if args.horizon_mode == "observed_targets"
        else "original_event_labels_retained_at_shifted_query_times; synthetic_horizon_test"
        if args.horizon_mode == "fixed_label_shift"
        else "original_facts_with_perturbed_test_timestamps",
        "noise_policy": "nested affected subsets; common noise across model seeds; no rounding/clipping; frozen split membership",
    }
    dump_json(destination / "result.json", result)
    print(json.dumps(result), flush=True)
