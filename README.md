# RelStream-ODE

**RelStream-ODE** models relational event streams using coupled continuous-time entity and relation dynamics. Relation representations respond to active ordered actor pairs and parameterize adaptive low-rank message transport. The evolved states are scored using a TuckER-style decoder for temporal knowledge graph forecasting.

This repository provides model training, evaluation, component ablations, temporal stress tests, and a common-protocol score evaluator for external baselines.

## Installation

Python 3.10 is recommended.

```bash
conda env create -f environment.yml
conda activate relstream-ode
python -m pip install -e .
```

Or install the dependencies using pip:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

Run the tests:

```bash
python -m pip install pytest
python -m pytest -q tests
```

## Datasets

Supported datasets: `ICEWS05-15`, `ICEWS14`, `ICEWS18`, and `GDELT`.

Obtain the processed chronological data files from the [RE-GCN project](https://github.com/Lee-zix/RE-GCN) in accordance with the original dataset licenses. Place them under a directory with the following layout:

```text
data-root/
├── ICEWS05-15/
├── ICEWS14/
├── ICEWS18/
└── GDELT/
```

Each dataset directory contains `stat.txt`, `train.txt`, `valid.txt`, and `test.txt`. See [data/README.md](data/README.md) for the record format and temporal units. You can supply a single dataset at a time.

```bash
export RELSTREAM_DATA_ROOT=/path/to/data-root
python scripts/prepare_data.py --datasets ICEWS14
```

## Training and evaluation

```bash
# Train using the default configuration
python train.py --dataset ICEWS14 --seed 42 --device cuda:0

# Resume a training run
python train.py --dataset ICEWS14 --seed 42 --resume --device cuda:0

# Evaluate the checkpoint selected on validation data
python evaluate.py \
  --checkpoint outputs/runs/full/ICEWS14/seed_42/best.pt \
  --device cuda:0
```

A short CPU execution check is available with `--smoke`:

```bash
python train.py --dataset ICEWS14 --seed 42 --device cpu --smoke
```

This runs a bounded diagnostic; it is not a full benchmark run. By default, completed runs are stored under `outputs/`. Training uses validation MRR for checkpoint selection and records run settings, data fingerprints, training metrics, and evaluation ranks.

## Ablations and experiments

JSON experiment configurations are provided in [`configs/`](configs/):

| Configuration | Description |
| --- | --- |
| `full` | Complete model |
| `no_relation_evolution` | Freeze relation dynamics |
| `identity_transport` | Replace the learned transport with identity |
| `joint_ablation` | Freeze relation dynamics and use identity transport |
| `no_graph_flow` | Remove graph-conditioned entity messages |
| `no_vector_gate` | Disable vector gating |
| `no_scalar_compatibility` | Use unit scalar message coefficients |
| `static_transport_coefficients` | Use time-invariant relation transport coefficients |
| `no_usage_summary` | Remove active-pair usage summaries |
| `max_usage` | Use max-pooling for relation usage |
| `attention_usage` | Use relation-conditioned attention aggregation |

```bash
python train.py \
  --dataset ICEWS14 --seed 42 \
  --config configs/joint_ablation.json

# Five independent random seeds
python scripts/run_experiments.py \
  --datasets ICEWS14 \
  --variants full no_relation_evolution identity_transport joint_ablation \
  --workers 1

python scripts/summarize.py
```

For parameter studies, edit a copy of `configs/full.json`. Configurable parameters include embedding dimension, transport rank, dropout, learning rate, `rk4_substeps`, `usage_aggregation`, and `transport_implementation` (`dense` or `factorized`).

## Temporal evaluations

```bash
# Evaluate future observed event targets at different leads
python scripts/run_stress.py \
  --checkpoint outputs/runs/full/GDELT/seed_42/best.pt \
  --experiment horizons --horizon-mode observed_targets \
  --offset-unit steps --offsets 10 20 30 40 50

# Perturb model time features while keeping event availability and labels fixed
python scripts/run_stress.py \
  --checkpoint outputs/runs/full/GDELT/seed_42/best.pt \
  --experiment jitter --noise-std-steps 3 \
  --noise-granularity timestamps --rates 20 30 40 50 60
```

The timestamp perturbation procedure preserves the original causal history and correct-answer filtering sets. The `events` noise granularity is also available but may be computationally expensive on large test sets.

## Baseline evaluation

The `scripts/evaluate_baseline_scores.py` interface recomputes tail-only filtered ranking metrics from **unfiltered full-entity score exports** of external models, including RE-GCN and TiRGN. Their separate training implementations are not distributed here.

Refer to [the score export and evaluation protocol](docs/BASELINE_PROTOCOL.md) for the required format and a command example. Matched evaluation also requires consistent training history, chronological splits, and model selection.

## Reproducibility

- The supported training seeds are `42`–`46`.
- Chronological split metadata are included in `splits/`; dataset files are obtained separately.
- `scripts/profile_resources.py` measures time and GPU memory for a saved checkpoint.
- The default solver is fixed-step RK4, with a configurable number of substeps.
- Source versions, configuration, dataset checksums, and per-query ranks are retained with experiment outputs.
- Load PyTorch checkpoints only from trusted sources.

The repository contains source and evaluation tools; it does not distribute benchmark datasets or pretrained checkpoints.
