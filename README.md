# RelStream-ODE

Coupled relation dynamics and adaptive information transport for temporal event forecasting. This repository includes the full model, three component ablations, repeated-seed training, independent evaluation, temporal stress experiments and resource measurements.

## Layout

```text
train.py / evaluate.py           Training and independent evaluation
src/relstream_ode/
  encoder.py                    Joint R-GCN initialization
  dynamics.py                   Actor-pair relation flow and typed transport
  solver.py                     Three-substep joint RK4
  decoder.py / model.py          TuckER scoring and complete model
  data.py / graph.py             Chronology, edge sets and causal histories
  trainer.py / evaluation.py     Optimization, recovery and filtered ranking
  experiments.py                Parallel seed/ablation queue
  statistics.py / analysis.py    Tables, paired intervals and figures
  stress.py / benchmark.py       Temporal stress and resource measurement
  config.py / utils.py           Settings, random states and atomic artifacts
configs/                        Full model and three ablation configurations
splits/                         Released input fingerprints and time ranges
scripts/                        Data, experiment, analysis and profiling commands
data/README.md                  External data layout
requirements.txt / environment.yml
```

## Environment and data

Python 3.10, PyTorch 2.4.0, NumPy 1.26.4, SciPy 1.12.0 and Matplotlib 3.9.2 are specified. DGL and an external ODE solver are not required.

```bash
conda env create -f environment.yml
conda activate relstream-ode
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export RELSTREAM_DATA_ROOT=/path/to/RE-GCN/data
python scripts/prepare_data.py
```

Alternatively install `requirements.txt` in a Python 3.10 environment. Run commands from this repository root. Source installation with `pip install -e .` is optional; the command entries work directly. Pass `--device cpu` for CPU execution.

Data are the processed chronological splits from [RE-GCN](https://github.com/Lee-zix/RE-GCN). See [data/README.md](data/README.md). Preparation verifies input checksums, vocabulary and row counts, then writes caches without changing the raw files.

| Dataset | Entities | Relations | Train | Validation | Test | Raw units / day |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ICEWS05-15 | 10,488 | 251 | 368,868 | 46,302 | 46,159 | 1 |
| ICEWS14 | 7,128 | 230 | 74,845 | 8,514 | 7,371 | 24 |
| ICEWS18 | 23,033 | 256 | 373,018 | 45,995 | 49,545 | 24 |
| GDELT | 7,691 | 240 | 1,734,399 | 238,765 | 305,241 | 1,440 |

Only the first four fact columns are used. The time coordinate is `(raw_time - first_raw_time) / raw_units_per_day`. GDELT bins are 15 minutes, so a snapshot step is 1/96 day. Window and horizon lengths in days use physical elapsed time.

## Native event exports

Use `scripts/convert_native.py` to train on new [ICEWS Coded Event Data](https://doi.org/10.7910/DVN/28075) or [GDELT 2.0 Events](https://gdeltproject.org/data.html) exports. Pass local event files and two UTC cutoffs. This example uses January through March for training, April for validation, and May onward for testing within the files supplied.

```bash
python scripts/convert_native.py --provider icews \
  --inputs /path/to/icews/events.2018.tab \
  --train-until 2018-04-01 --valid-until 2018-05-01 \
  --output data/ICEWS-native

python train.py --dataset ICEWS-native --data-root data --seed 42
python evaluate.py --data-root data \
  --checkpoint outputs/runs/full/ICEWS-native/seed_42/best.pt
```

For GDELT, pass `--provider gdelt2` and the Events `*.export.CSV.zip` files for the desired period. The converter uses UTC `DATEADDED`, which records when GDELT added the event. ICEWS uses its original daily `Event Date`. A conversion emits chronological quadruples, training-derived actor/relation vocabularies, and a manifest documenting input hashes and excluded rows. See the [native file formats and conversion rules](data/README.md#native-exports).

The four benchmark names remain reserved for the released splits. Use a distinct name for native data and the single-run training and evaluation commands above. The benchmark queue and tables cover the four released datasets.

## Train and evaluate

```bash
python train.py --dataset ICEWS14 --seed 42
python train.py --dataset ICEWS14 --seed 42 --resume
python evaluate.py --checkpoint outputs/runs/full/ICEWS14/seed_42/best.pt

python train.py --dataset ICEWS14 --seed 42 --config configs/no_relation_evolution.json
python train.py --dataset ICEWS14 --seed 42 --config configs/identity_transport.json
python train.py --dataset ICEWS14 --seed 42 --config configs/no_graph_flow.json
```

`full` evolves both branches. `no_relation_evolution` sets the relation derivative to zero after initialization. `identity_transport` replaces the typed transport operator with identity while retaining relation flow. `no_graph_flow` sets the entity derivative to zero while retaining relation flow and TuckER. Disabled modules are excluded from trainable-parameter counts.

Each run writes `config.json`, `protocol.json`, `status.json`, `epochs.json`, `best.pt`, `last.pt`, `result.json` and `test_ranks.npz` under `outputs/runs/<variant>/<dataset>/seed_<seed>/`. Recovery preserves model and BatchNorm buffers, Adam, StepLR, all random states, timestamp order, optimizer steps and partial-epoch statistics. SIGINT/SIGTERM requests save at the next timestamp boundary. Source/configuration/data/runtime mismatches reject recovery.

For a bounded execution check:

```bash
python train.py --dataset ICEWS14 --seed 42 --device cpu --smoke
```

This keeps the complete vocabulary and model widths, uses two training/validation/test timestamps, at most 32 queries per timestamp, up to 1,024 graph rows and a one-bin history. It writes to `outputs/diagnostic_runs/` and is excluded from experiment summaries. Evaluation of its checkpoint automatically retains these limits; `--full-split` evaluates every query while preserving the checkpoint's declared diagnostic configuration.

## Repeated runs

```bash
# Four variants × four datasets × seeds 42–46: 80 runs.
python scripts/run_experiments.py --workers 4

# A smaller campaign.
python scripts/run_experiments.py --workers 2 --datasets ICEWS14 --variants full
python scripts/summarize.py
python scripts/plot_results.py
```

The queue skips compatible completed runs, resumes interruptions, retains per-job logs and stops its remaining child processes after a failure. `--workers` controls the number of independent processes. Configure or select hyperparameters with `--config <file> --validation-only --output-root outputs/selection/<name>`; this stops after validation selection without evaluating test. Use a fresh output root for the final fixed configuration.

`outputs/tables/` contains per-seed MRR/Hits, best epoch, validation MRR, checkpoint references, mean/sample-SD/range tables, matched differences and execution status. Metrics are stored as fractions; multiply by 100 for percentages. Figures use complete five-seed cohorts. Use the same `--output-root` for every command in a campaign.

## Model and evaluation settings

The default widths are entity/relation 200 and time 64, with transport rank 16. Two joint R-GCN layers initialize states from the cumulative graph at the first boundary. Initialization uses incoming relation-normalized neighbors; subsequent entity flow aggregates outgoing typed edges. Ordered actor pairs are deduplicated within each interval and their representations are averaged separately by relation. Relation summaries and transport operators are recomputed at every RK4 stage from the same intermediate joint state. Compatibility is a signed dot product, and entity-derivative LayerNorm has no learnable affine parameters.

The default history window is 100 elapsed days, starting no earlier than the first available observation. Pair, relation-flow and transport MLP hidden widths are 200. Time encoding uses alternating sine/cosine features with frequencies `10000^(-2q/64)`. These settings, Xavier initialization and decoder normalization are explicit implementation settings. Decoder BatchNorm normalizes the complete entity and relation state tables before dropout; the resulting entity table is shared by query and candidate scoring. The transport matrices are materialized densely for active relations.

Adam uses learning rate 0.001 and weight decay 1e-5. Batch size is 32, dropout 0.3 and gradient clipping 1. The objective is mean cross-entropy over the complete entity vocabulary, with no negative sampling. Training runs all 100 epochs without patience-based stopping. StepLR reduces the learning rate by 0.1 after every 30 epochs. Only validation MRR selects checkpoints; the earliest checkpoint is retained for equal MRR. The first training timestamp supplies history rather than a forecast target. Target timestamps and their rows are shuffled using saved random states.

Each interval uses three classical RK4 substeps, jointly advancing entities and relations. Differentiation is through the numerical operations, with no continuous adjoint. `--checkpoint-segments 8` recomputes deterministic groups of eight intervals during backward to reduce saved activations; use 0 to retain all intermediate activations. This does not truncate gradients or detach interval states.

Evaluation ranks original forward tail queries over every released entity. No inverse edges, reciprocal queries, popularity exclusions or additional static word graphs are used. Filtering removes other true tails with the same subject, relation and exact target timestamp, while retaining the target. Average rank is `1 + strictly_higher + 0.5 * (equal_including_target - 1)` and Hits@K uses rank ≤ K. `--tie optimistic` and `--tie pessimistic` provide explicit sensitivity settings. Histories use strictly earlier events; rolling evaluation can use previously observed validation/test events. Deterministic inference reuses a joint state only when parameters and the first initialization boundary are fixed; training always recomputes evolving representations after updates.

Repeated-run summaries report arithmetic mean and sample standard deviation. Five matched seeds give difference interval `mean(d) ± t(0.975,4) * sample_sd(d)/sqrt(5)`, conditional on the fixed splits/protocol and approximately independent, normally distributed seed differences. Exact two-sided sign flips assume symmetry and enumerate all 32 assignments; five nonzero pairs have minimum p = 0.0625. Missing or incompatible cohorts are withheld. These exploratory comparisons have limited power with five seeds.

## Temporal stress experiments

```bash
python scripts/run_stress.py --checkpoint outputs/runs/full/GDELT/seed_42/best.pt \
  --experiment jitter --rates 20 30 40 50 60 --noise-std-days 3

python scripts/run_stress.py --checkpoint outputs/runs/full/GDELT/seed_42/best.pt \
  --experiment horizons --horizon-mode observed_targets --offset-unit days \
  --offsets 10 20
```

GDELT spans about 31 actual days. `observed_targets` preserves observed target times and moves the history cutoff back by each lead time. It evaluates the common target cohort eligible at every requested offset. A 40/50-day lead time has no past observations in these inputs, so requesting 10–50 actual days writes `not_evaluable` records rather than numeric scores. `--offset-unit snapshots` explicitly measures offsets in 15-minute steps.

`--horizon-mode fixed_label_shift` is a separate synthetic fixed-history experiment: for each original query, freeze history at its immediately preceding observation boundary, retain its original event label and filtering group, and move the query time to that boundary plus the offset. These shifted labels are not observations of new future events. Its outputs state this convention and preserve original labels alongside shifted times. Choose the target-label convention explicitly when running a horizon experiment. All non-diagnostic stress runs use 100-day input windows, clipped to available history.

Jitter selects the requested fraction of test rows after splitting and adds independent zero-mean Gaussian noise in days. Entity/relation IDs and split membership stay fixed; histories and filtering are rebuilt from the perturbed times. No timestamp rounding or clipping is applied. Noise seed 2026 and nested affected subsets provide the same disturbances across model seeds and variants. Assignments are saved. Stress figures require five complete matching seeds. Add `--smoke` for bounded diagnostics, which use a one-bin history and remain outside formal summaries.

## Resource measurement

After other GPU jobs finish:

```bash
python scripts/profile_resources.py --checkpoint outputs/runs/full/ICEWS14/seed_42/best.pt
python scripts/summarize.py
```

The serial profiler measures one representative training epoch from selected weights with fresh Adam, then restores the selected model for three complete validation and test evaluations. Training warm-up performs forward/backward and restores weights/random states; inference warm-up covers three timestamps. CUDA is synchronized around every timed phase. Allocated and reserved PyTorch allocator peaks are reset and recorded in bytes; these exclude CUDA context and allocations outside PyTorch. The restoration copy remains on CPU.

Training timing includes graph construction, transfer, encoding, scoring, cross-entropy, backward, clipping and Adam. Inference includes history construction, encoding, full-entity scoring and filtering. Initial data preparation, checkpoint loading, warm-up and artifact I/O are excluded. Per-query latency divides complete-test time by the measured query count. Raw repetitions, counts, arithmetic mean, sample SD and range are retained. CPU measurements are labeled CPU and report zero CUDA allocator bytes. Actual epoch/validation times and end-to-end training wall time, including validation and checkpoint I/O, are also retained in each training run; a representative post-selection epoch is a separate measurement.

## License

The repository software and its documentation are available under the [MIT License](LICENSE). Obtain event data from their respective providers and follow the applicable terms for each dataset.
