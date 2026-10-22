"""Figures from complete measured seed cohorts."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import ROOT, SEEDS
from .statistics import write_csv
from .utils import dump_json, fingerprint


def main():
    parser = argparse.ArgumentParser(
        description="Plot component comparisons and temporal stress measurements"
    )
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs")
    args = parser.parse_args()
    root = args.output_root
    figures = root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    stress = {}
    for path in (root / "stress").glob("*/*/seed_*/*/result.json"):
        record = json.loads(path.read_text())
        if record["diagnostic"]:
            continue
        settings = record["settings"]
        for row in record["rows"]:
            if row["status"] != "completed":
                continue
            parameter = row.get("offset", row.get("rate_percent"))
            setting_id = fingerprint(
                {
                    "settings": settings,
                    "config": record["model_config_fingerprint"],
                    "protocol": record["protocol_fingerprint"],
                    "data": record["data_fingerprint"],
                    "source": record["source_sha256"],
                }
            )[:12]
            key = (
                record["variant"],
                record["dataset"],
                settings["experiment"],
                settings["horizon_mode"],
                settings["offset_unit"],
                setting_id,
                parameter,
            )
            group = stress.setdefault(key, {})
            if record["seed"] in group:
                raise ValueError(
                    "Duplicate stress seed/setting; use separate output roots"
                )
            group[record["seed"]] = row
    rows = []
    series = {}
    for key, seeds in stress.items():
        if set(seeds) != set(SEEDS):
            continue
        variant, dataset, experiment, mode, unit, setting_id, parameter = key
        counts = {r["queries"] for r in seeds.values()}
        if len(counts) != 1:
            raise ValueError("Stress seeds do not evaluate the same query cohort")
        values = np.asarray([seeds[s]["MRR"] for s in SEEDS]) * 100
        row = {
            "variant": variant,
            "dataset": dataset,
            "experiment": experiment,
            "horizon_mode": mode,
            "unit": unit,
            "setting_id": setting_id,
            "parameter": parameter,
            "n": 5,
            "MRR_percent_mean": float(values.mean()),
            "MRR_percent_sd": float(values.std(ddof=1)),
            "queries": next(iter(counts)),
        }
        rows.append(row)
        series.setdefault(key[:-1], []).append(row)
    for key, values in series.items():
        values = sorted(values, key=lambda row: row["parameter"])
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.errorbar(
            [r["parameter"] for r in values],
            [r["MRR_percent_mean"] for r in values],
            yerr=[r["MRR_percent_sd"] for r in values],
            marker="o",
            capsize=3,
        )
        ax.set_xlabel(
            "Perturbed units (%)"
            if key[2] == "jitter"
            else f"Target offset ({key[4]})"
        )
        ax.set_ylabel("Filtered MRR (%)")
        ax.set_title(key[0] + " / " + key[1])
        fig.tight_layout()
        filename = "_".join(str(x) for x in key if x is not None)
        fig.savefig(figures / f"{filename}.png", dpi=200)
        fig.savefig(figures / f"{filename}.pdf")
        plt.close(fig)
    tables = root / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    write_csv(
        tables / "stress_summary.csv",
        rows,
        [
            "variant",
            "dataset",
            "experiment",
            "horizon_mode",
            "unit",
            "setting_id",
            "parameter",
            "n",
            "MRR_percent_mean",
            "MRR_percent_sd",
            "queries",
        ],
    )
    component_groups = {}
    summary = tables / "model_summary.csv"
    if summary.exists():
        with summary.open() as stream:
            for row in csv.DictReader(stream):
                if row["metric"] == "MRR" and int(row["n"]) == 5:
                    component_groups.setdefault(row["dataset"], {})[row["variant"]] = (
                        row
                    )
    component_figures = 0
    order = ["full", "no_relation_evolution", "identity_transport", "no_graph_flow"]
    for dataset, values in component_groups.items():
        if not set(order).issubset(values):
            continue
        means = np.asarray([float(values[k]["mean"]) for k in order]) * 100
        sd = np.asarray([float(values[k]["sample_sd"]) for k in order]) * 100
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(range(4), means, yerr=sd, capsize=3)
        ax.set_xticks(
            range(4),
            ["Full", "No relation\nevolution", "Identity\ntransport", "No graph\nflow"],
        )
        ax.set_ylabel("Filtered MRR (%)")
        ax.set_title(dataset)
        fig.tight_layout()
        fig.savefig(figures / f"components_{dataset}.png", dpi=200)
        fig.savefig(figures / f"components_{dataset}.pdf")
        plt.close(fig)
        component_figures += 1
    dump_json(
        figures / "status.json",
        {
            "complete_five_seed_stress_series": len(series),
            "component_figures": component_figures,
            "measured_rows": len(rows),
            "missing_seed_cohorts_omitted": True,
        },
    )
