"""Explicit model, optimization and temporal settings."""

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("RELSTREAM_DATA_ROOT", ROOT / "data"))
DATASETS = ("ICEWS05-15", "ICEWS14", "ICEWS18", "GDELT")
VARIANTS = ("full", "no_relation_evolution", "identity_transport", "no_graph_flow")
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
    "history_days": 100.0,
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
    ]:
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if config["entity_dim"] < 8 or config["time_dim"] % 2:
        raise ValueError("entity_dim must be at least 8 and time_dim must be even")
    if config["transport_rank"] > config["entity_dim"]:
        raise ValueError("Transport rank cannot exceed entity width")
    for key in ["history_days", "layer_norm_epsilon", "learning_rate", "gradient_clip"]:
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
    if config["rk4_substeps"] != 3:
        raise ValueError(
            "The supported integration rule uses three classical RK4 substeps"
        )
    if config["variant"] not in VARIANTS:
        raise ValueError("Unknown model variant")


def source_hashes():
    from .utils import sha256

    return {
        str(p.relative_to(ROOT)): sha256(p)
        for p in sorted((ROOT / "src/relstream_ode").glob("*.py"))
    }
