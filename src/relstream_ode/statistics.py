"""Actual per-seed metrics, complete matched comparisons and resource tables."""

import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t

from .config import DATASETS, ROOT, SEEDS, VARIANTS
from .utils import dump_json

METRICS = ["MRR", "Hits@1", "Hits@3", "Hits@10"]


def write_csv(path, rows, columns):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def paired_interval(differences):
    differences = np.asarray(differences, dtype=np.float64)
    if differences.shape != (5,) or not np.isfinite(differences).all():
        raise ValueError("Five finite matched differences are required")
    mean = float(differences.mean())
    sd = float(differences.std(ddof=1))
    half = float(student_t.ppf(0.975, 4)) * sd / np.sqrt(5)
    candidates = [
        abs(np.mean(differences * np.asarray(signs)))
        for signs in itertools.product([-1, 1], repeat=5)
    ]
    p = float(np.mean(np.asarray(candidates) >= abs(mean) - 1e-15))
    return {
        "difference_mean": mean,
        "difference_sd": sd,
        "difference_ci95_low": mean - half,
        "difference_ci95_high": mean + half,
        "sign_flip_p_exploratory": p,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Summarize completed full runs without substituting reference-table values"
    )
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs")
    args = parser.parse_args()
    root = args.output_root.resolve()
    out = root / "tables"
    out.mkdir(parents=True, exist_ok=True)
    records = []
    groups = {}
    status = []
    diagnostic = []
    for variant, dataset, seed in itertools.product(VARIANTS, DATASETS, SEEDS):
        folder = root / "runs" / variant / dataset / f"seed_{seed}"
        state = (
            json.loads((folder / "status.json").read_text())
            if (folder / "status.json").exists()
            else {"status": "not_started"}
        )
        status.append(
            {
                "variant": variant,
                "dataset": dataset,
                "seed": seed,
                "status": state["status"],
            }
        )
        path = folder / "result.json"
        if not path.exists():
            continue
        r = json.loads(path.read_text())
        if (
            r["status"] != "completed"
            or not r.get("full_test")
            or not r.get("full_training")
            or r.get("diagnostic")
        ):
            continue
        r["config"] = str((folder / "config.json").relative_to(root))
        r["checkpoint"] = str((folder / "best.pt").relative_to(root))
        records.append(r)
        groups.setdefault((variant, dataset), {})[seed] = (
            r,
            json.loads((folder / "config.json").read_text()),
        )
    for path in (root / "diagnostic_runs").glob("*/*/seed_*/result.json"):
        r = json.loads(path.read_text())
        r["result_file"] = str(path.relative_to(root))
        diagnostic.append(r)
    summaries = []
    pairs = []
    for (variant, dataset), seeds in groups.items():
        configs = {json.dumps(x[1], sort_keys=True) for x in seeds.values()}
        if len(configs) != 1 or any(
            len({x[0][key] for x in seeds.values()}) != 1
            for key in ["data_fingerprint", "protocol_fingerprint"]
        ):
            raise ValueError(
                "Repeated seeds have different settings or data/evaluation protocols"
            )
        if (
            len(
                {
                    json.dumps(x[0]["source_sha256"], sort_keys=True)
                    for x in seeds.values()
                }
            )
            != 1
        ):
            raise ValueError("Repeated seeds have different executable sources")
        for metric in METRICS:
            values = np.asarray([x[0][metric] for x in seeds.values()])
            summaries.append(
                {
                    "variant": variant,
                    "dataset": dataset,
                    "metric": metric,
                    "n": len(values),
                    "mean": float(values.mean()),
                    "sample_sd": float(values.std(ddof=1)) if len(values) > 1 else None,
                    "minimum": float(values.min()),
                    "maximum": float(values.max()),
                }
            )
    for dataset in DATASETS:
        full = groups.get(("full", dataset), {})
        for variant in VARIANTS[1:]:
            other = groups.get((variant, dataset), {})
            if set(full) != set(SEEDS) or set(other) != set(SEEDS):
                continue
            for seed in SEEDS:
                a, ca = full[seed]
                b, cb = other[seed]
                if {k: v for k, v in ca.items() if k != "variant"} != {
                    k: v for k, v in cb.items() if k != "variant"
                } or any(
                    a[key] != b[key]
                    for key in [
                        "data_fingerprint",
                        "protocol_fingerprint",
                        "source_sha256",
                    ]
                ):
                    raise ValueError(
                        "Matched models differ in settings beyond the specified ablation"
                    )
            for metric in METRICS:
                differences = [full[s][0][metric] - other[s][0][metric] for s in SEEDS]
                pairs.append(
                    {
                        "dataset": dataset,
                        "comparison": "full - " + variant,
                        "metric": metric,
                        **{
                            f"difference_seed_{s}": d
                            for s, d in zip(SEEDS, differences)
                        },
                        **paired_interval(differences),
                    }
                )
    write_csv(
        out / "per_seed_results.csv",
        records,
        [
            "variant",
            "dataset",
            "seed",
            *METRICS,
            "best_epoch",
            "validation_MRR",
            "queries",
            "epochs_completed",
            "config",
            "checkpoint",
            "checkpoint_sha256",
            "data_fingerprint",
            "protocol_fingerprint",
        ],
    )
    write_csv(
        out / "model_summary.csv",
        summaries,
        [
            "variant",
            "dataset",
            "metric",
            "n",
            "mean",
            "sample_sd",
            "minimum",
            "maximum",
        ],
    )
    write_csv(
        out / "paired_differences.csv",
        pairs,
        [
            "dataset",
            "comparison",
            "metric",
            *[f"difference_seed_{s}" for s in SEEDS],
            "difference_mean",
            "difference_sd",
            "difference_ci95_low",
            "difference_ci95_high",
            "sign_flip_p_exploratory",
        ],
    )
    write_csv(
        out / "experiment_status.csv", status, ["variant", "dataset", "seed", "status"]
    )
    write_csv(
        out / "diagnostic_results.csv",
        diagnostic,
        [
            "variant",
            "dataset",
            "seed",
            *METRICS,
            "best_epoch",
            "validation_MRR",
            "queries",
            "diagnostic",
            "result_file",
        ],
    )
    observed_resources = []
    for record in records:
        folder = root / Path(record["config"]).parent
        epochs = json.loads((folder / "epochs.json").read_text())
        measurements = {
            "training_seconds_per_epoch": [e["training_seconds"] for e in epochs],
            "validation_seconds_per_epoch": [e["validation_seconds"] for e in epochs],
            "end_to_end_training_hours": [record["training_wall_seconds"] / 3600],
            "peak_allocated_bytes": [e["peak_allocated_bytes"] for e in epochs],
            "peak_reserved_bytes": [e["peak_reserved_bytes"] for e in epochs],
        }
        for name, raw in measurements.items():
            values = np.asarray(raw, dtype=float)
            observed_resources.append(
                {
                    "variant": record["variant"],
                    "dataset": record["dataset"],
                    "seed": record["seed"],
                    "measurement": name,
                    "n": len(values),
                    "mean": float(values.mean()),
                    "sample_sd": float(values.std(ddof=1)) if len(values) > 1 else None,
                    "minimum": float(values.min()),
                    "maximum": float(values.max()),
                    "device": record["hardware"]["device"],
                    "record": str((folder / "result.json").relative_to(root)),
                }
            )
    write_csv(
        out / "resources_observed_training.csv",
        observed_resources,
        [
            "variant",
            "dataset",
            "seed",
            "measurement",
            "n",
            "mean",
            "sample_sd",
            "minimum",
            "maximum",
            "device",
            "record",
        ],
    )
    resource_rows = []
    for path in (root / "resources").glob("*/*/seed_*/result.json"):
        result = json.loads(path.read_text())
        if result.get("diagnostic"):
            continue
        for name in [
            "training_seconds_per_epoch",
            "validation_seconds",
            "full_test_seconds",
            "inference_ms_per_query",
            "peak_allocated_bytes",
            "peak_reserved_bytes",
        ]:
            values = np.asarray(result[name], dtype=float)
            resource_rows.append(
                {
                    "variant": result["variant"],
                    "dataset": result["dataset"],
                    "seed": result["seed"],
                    "measurement": name,
                    "n": len(values),
                    "mean": float(values.mean()),
                    "sample_sd": float(values.std(ddof=1)) if len(values) > 1 else None,
                    "minimum": float(values.min()),
                    "maximum": float(values.max()),
                    "trainable_parameters": result["trainable_parameters"],
                    "device": result["hardware"]["device"],
                    "record": str(path.relative_to(root)),
                }
            )
    write_csv(
        out / "resources.csv",
        resource_rows,
        [
            "variant",
            "dataset",
            "seed",
            "measurement",
            "n",
            "mean",
            "sample_sd",
            "minimum",
            "maximum",
            "trainable_parameters",
            "device",
            "record",
        ],
    )
    dump_json(
        out / "summary_status.json",
        {
            "completed_full_runs": len(records),
            "expected_runs": len(DATASETS) * len(VARIANTS) * len(SEEDS),
            "paired_rows": len(pairs),
            "diagnostic_runs": len(diagnostic),
        },
    )
    print(
        json.dumps(
            {
                "completed_full_runs": len(records),
                "paired_rows": len(pairs),
                "tables": str(out),
            }
        )
    )
