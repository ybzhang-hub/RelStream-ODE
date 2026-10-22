"""Independent rank evaluator for external full-vocabulary baseline scores.

This does not train RE-GCN/TiRGN. Their code must export scores in the
contract stated in docs/BASELINE_PROTOCOL.md, without post-filtering.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .config import DATA_ROOT, ROOT, DATASETS
from .data import TemporalDataset
from .evaluation import filtered_ranks, summarize_ranks
from .utils import dump_json, fingerprint, sha256


def evaluate_score_shards(data, split, scores_dir, tie="average"):
    """Validate exact query coverage then perform common original-time filtering."""
    scores_dir = Path(scores_dir)
    ranks, row_ids = [], []
    expected_files = set()
    for timestamp, truth_rows in data.split_times[split].items():
        from_index = int(np.searchsorted(data.times[split], timestamp, side="left"))
        count = len(truth_rows)
        original_ids = data.row_ids[split][from_index:from_index + count]
        raw_ts = int(round(timestamp * data.units + data.origin))
        paths = sorted(scores_dir.glob(f"t_{raw_ts}_part_*.npz"))
        if not paths:
            raise FileNotFoundError(f"Missing exported scores for raw timestamp {raw_ts}")
        expected_files.update(paths)
        covered = set()
        answers = data.answers(timestamp)
        for path in paths:
            with np.load(path, allow_pickle=False) as shard:
                if set(shard.files) != {"row_ids", "scores"}:
                    raise ValueError(f"Invalid shard keys in {path}")
                ids = np.asarray(shard["row_ids"], dtype=np.int64)
                scores = np.asarray(shard["scores"], dtype=np.float32)
            if scores.shape != (len(ids), data.entities) or len(ids) == 0:
                raise ValueError(f"Incorrect full-vocabulary score tensor in {path}")
            if not np.isfinite(scores).all():
                raise FloatingPointError(f"Nonfinite prediction scores: {path}")
            if len(np.unique(ids)) != len(ids):
                raise ValueError(f"Duplicate query IDs within {path}")
            if any(int(x) in covered for x in ids):
                raise ValueError(f"Duplicate query IDs across shards: {path}")
            if not np.isin(ids, original_ids).all():
                raise ValueError(f"Foreign query ID at timestamp {raw_ts}: {path}")
            covered.update(int(x) for x in ids)
            queries = np.asarray(data.raw[split][ids, :3], dtype=np.int64)
            rank = filtered_ranks(
                torch.from_numpy(scores), torch.from_numpy(queries), answers, tie
            ).numpy()
            row_ids.extend(ids.tolist()); ranks.extend(rank.tolist())
        if covered != set(original_ids.tolist()):
            raise ValueError(f"Missing query IDs at timestamp {raw_ts}")
    all_npz = set(scores_dir.glob("*.npz"))
    if all_npz != expected_files:
        raise ValueError(f"Unknown/stale score shards: {sorted(all_npz - expected_files)}")
    return summarize_ranks(ranks), {
        "row_ids": np.asarray(row_ids, dtype=np.int64),
        "ranks": np.asarray(ranks, dtype=np.float64),
    }, {p.name: sha256(p) for p in sorted(expected_files)}


def main():
    parser = argparse.ArgumentParser(
        description="Validate externally exported baseline scores under RelStream-ODE ranking protocol"
    )
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--split", choices=["valid", "test"], default="test")
    parser.add_argument("--scores-dir", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--cache-root", type=Path, default=ROOT / "cache")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tie", default="average", choices=["average", "optimistic", "pessimistic"])
    args = parser.parse_args()
    metadata = json.loads(args.metadata.read_text())
    for key in ["baseline", "source_commit", "seed", "training_protocol", "selected_on_validation"]:
        if key not in metadata:
            raise ValueError(f"Missing mandatory baseline provenance field {key}")
    if metadata["selected_on_validation"] is not True:
        raise ValueError("Baseline hyperparameters must be validation-selected")
    data = TemporalDataset(args.dataset, args.data_root, args.cache_root)
    result, record, hashes = evaluate_score_shards(data, args.split, args.scores_dir, args.tie)
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / "ranks.npz", **record)
    payload = {
        "status": "completed_external_scores_only",
        "dataset": args.dataset, "split": args.split,
        "baseline": metadata, "baseline_metadata_hash": fingerprint(metadata),
        "score_file_sha256": hashes,
        "data_fingerprint": data.fingerprint,
        "protocol": {
            "direction": "tail_only", "candidate_set": "entire_released_entity_vocabulary",
            "filter_truth": "all_correct_tails_at_original_target_time",
            "tie": args.tie,
        },
        **result,
    }
    dump_json(args.output / "result.json", payload)
    print(json.dumps({"baseline": metadata["baseline"], **result}), flush=True)


if __name__ == "__main__":
    main()
