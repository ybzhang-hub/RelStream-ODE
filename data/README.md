# Benchmark datasets

Processed chronological temporal knowledge graph splits are available from the [RE-GCN repository](https://github.com/Lee-zix/RE-GCN). Observe the data providers' terms of use.

Each dataset directory contains:

```text
<data-root>/
  ICEWS14/
    stat.txt
    train.txt
    valid.txt
    test.txt
```

`stat.txt` supplies entity and relation counts. The training, validation, and test files contain rows of the form `subject relation object timestamp`. IDs are integers, and query histories use events with timestamps strictly earlier than the target time.

| Dataset | Raw units per day | Raw units per model time step |
| --- | ---: | ---: |
| ICEWS05-15 | 1 | 1 |
| ICEWS14 | 24 | 24 |
| ICEWS18 | 24 | 24 |
| GDELT | 1,440 | 15 |

The default model clock is `(raw_timestamp - origin) / raw_units_per_model_step`. Preprocessing verifies expected dataset statistics, chronological splitting, and file hashes without editing source data.

```bash
export RELSTREAM_DATA_ROOT=/path/to/data-root
python scripts/prepare_data.py --datasets ICEWS14
```

Cached arrays and other generated artifacts are written to `cache/` and `outputs/`, which are excluded from version control.
