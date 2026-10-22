"""Explicit model, optimization and temporal settings."""

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("RELSTREAM_DATA_ROOT", ROOT / "data"))
DATASETS = ("ICEWS05-15", "ICEWS14", "ICEWS18", "GDELT")
CORE_VARIANTS = ("full", "no_relation_evolution", "identity_transport", "no_graph_flow")
VARIANTS = (
    "full", "no_relation_evolution", "identity_transport", "no_graph_flow",
    "joint_ablation", "no_vector_gate", "no_scalar_compatibility",
    "static_transport_coefficients", "no_usage_summary",
    "max_usage", "attention_usage",
)
SEEDS = (42, 43, 44, 45, 46)
DEFAULTS = {
    "entity_dim": 200,
    "relation_dim": 200,
    "time_dim": 64,
    "transport_rank": 16,
    "initializer_layers": 2,
    "pair_hidden_dim": 200,
    "relation_hidden_dim": 200,
    "transport_hidden_dim": 200,
    "dropout": 0.3,
    "history_steps": 100.0,
    "time_coordinate": "benchmark_steps",
    "transport_formula": "paper",
    "transport_implementation": "dense",
    "usage_aggregation": "mean",
    "rk4_substeps": 3,
    "layer_norm_epsilon": 1e-5,
    "time_frequency_base": 10000.0,
    "learning_rate": 1e-3,
    "weight_decay": 1e-5,
    "batch_size": 32,
    "epochs": 100,
    "scheduler_step": 30,
    "scheduler_gamma": 0.1,
    "gradient_clip": 1.0,
    "variant": "full",
}


def load_config(path=None):
    config = dict(DEFAULTS)
    if path is not None:
        supplied = json.loads(Path(path).read_text())
        unknown = set(supplied) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
        config.update(supplied)
    validate_config(config)
    return config


def validate_config(config):
    if set(config) != set(DEFAULTS):
        raise ValueError("Configuration fields must match the supported schema")
    for key in [
        "entity_dim",
        "relation_dim",
        "time_dim",
        "transport_rank",
        "initializer_layers",
        "pair_hidden_dim",
        "relation_hidden_dim",
        "transport_hidden_dim",
        "batch_size",
        "epochs",
        "scheduler_step",
        "rk4_substeps",
    ]:
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if config["entity_dim"] < 8 or config["time_dim"] % 2:
        raise ValueError("entity_dim must be at least 8 and time_dim must be even")
    if config["transport_rank"] > config["entity_dim"]:
        raise ValueError("Transport rank cannot exceed entity width")
    for key in ["history_steps", "layer_norm_epsilon", "learning_rate", "gradient_clip"]:
        value = config[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not 0 < value < float("inf")
        ):
            raise ValueError(f"{key} must be finite and positive")
    if not 0 <= config["dropout"] < 1 or not 0 <= config["weight_decay"] < float("inf"):
        raise ValueError("Invalid dropout or weight decay")
    if not 0 < config["scheduler_gamma"] <= 1 or config["time_frequency_base"] <= 1:
        raise ValueError("Invalid scheduler gamma or time frequency base")
    if config["time_coordinate"] not in {"benchmark_steps", "legacy_days"}:
        raise ValueError("Unsupported time coordinate")
    if config["transport_formula"] not in {"paper", "legacy"}:
        raise ValueError("Unsupported transport formula")
    if config["transport_implementation"] not in {"dense", "factorized"}:
        raise ValueError("Unsupported transport implementation")
    if config["usage_aggregation"] not in {"mean", "max", "attention"}:
        raise ValueError("Unsupported relation-use aggregation")
    if config["variant"] not in VARIANTS:
        raise ValueError("Unknown model variant")


def source_hashes():
    from .utils import sha256

    return {
        str(p.relative_to(ROOT)): sha256(p)
        for p in sorted((ROOT / "src/relstream_ode").glob("*.py"))
    }


def checkpoint_config(config):
    """Migrate pre-correction checkpoints to their *actual* historical implementation."""
    config = dict(config)
    if "history_days" in config and "history_steps" not in config:
        config["history_steps"] = config.pop("history_days")
        config.setdefault("time_coordinate", "legacy_days")
        config.setdefault("transport_formula", "legacy")
        config.setdefault("transport_implementation", "dense")
        config.setdefault("usage_aggregation", "mean")
    for key in DEFAULTS:
        if key not in config:
            raise ValueError(f"Checkpoint is missing required configuration field: {key}")
    validate_config(config)
    return config
