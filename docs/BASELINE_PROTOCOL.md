# External baseline score export contract

This adapter can assess score exports from third-party RE-GCN and TiRGN
implementations under **one common ranking protocol**. It does not ensure the
external model used the same historical data or training setup. Those must
also be checked and reported separately.

1. Use exactly the released chronological split from `splits/<dataset>.json`.
2. Score original directed **tail** queries `(s,r,?,t)` without reciprocal heads.
3. Produce a raw score for **every** candidate ID in `[0, num_entities)`.
4. Do **not** apply candidate filtering to the exported scores.
5. Use only events observed strictly before the query's original time.
6. Select training hyperparameters and checkpoints on the validation split.
7. Export each timestamp in one or more `.npz` shards (no pickle):
   `t_<raw_integer_timestamp>_part_<number>.npz`, with exactly two arrays:
   `row_ids` (int64 original row indices into that split's `.txt` file), and
   `scores` (float32, shape `[len(row_ids), entities]`).
   Every original query appears exactly once across shards and there are no
   extra queries. Directory must contain no stale unrelated `.npz` files.
8. Supply a JSON file containing at least:

```json
{
  "baseline": "TiRGN",
  "source_commit": "REPLACE_WITH_ACTUAL_GIT_HASH",
  "seed": 42,
  "training_protocol": "tail_only; history strictly earlier than query",
  "selected_on_validation": true
}
```

9. Evaluate after exporting:

```bash
python scripts/evaluate_baseline_scores.py \
  --dataset ICEWS14 --split test --scores-dir exports/TiRGN/seed_42 \
  --metadata exports/TiRGN/seed_42/metadata.json \
  --output outputs/external/TiRGN/ICEWS14/seed_42
```

The evaluator rejects absent or duplicated query IDs, incorrect vocab width,
non-finite scores, and score files outside the requested split. It recomputes
time-aware filtered tail ranks using the **original** target time, stores per
query ranks and source score file hashes, and reports MRR/Hits. Retain original baseline logs/configs, their actual Git commits and five independent seeds. Running this adapter alone does not
establish end-to-end training-protocol equality.
