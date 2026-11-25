"""Released chronological event splits with a common day-based time axis."""

import argparse
import json
import os
import tempfile
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .config import DATA_ROOT, DATASETS, ROOT
from .graph import EventGraph
from .native_stream import NATIVE_FORMAT, SECONDS_PER_DAY, SPLITS, validate_dataset_name
from .utils import dump_json, fingerprint, sha256

SPECS = {
    "ICEWS05-15": (10488, 251, (368868, 46302, 46159), 1.0),
    "ICEWS14": (7128, 230, (74845, 8514, 7371), 24.0),
    "ICEWS18": (23033, 256, (373018, 45995, 49545), 24.0),
    "GDELT": (7691, 240, (1734399, 238765, 305241), 1440.0),
}


def locate_data(root):
    root = Path(root)
    for candidate in [root, root / "data"]:
        if all((candidate / name / "stat.txt").is_file() for name in DATASETS):
            return candidate
    raise FileNotFoundError(
        "Expected RE-GCN data/{ICEWS05-15,ICEWS14,ICEWS18,GDELT}; set RELSTREAM_DATA_ROOT or --data-root"
    )


def dataset_directory(root, dataset):
    validate_dataset_name(dataset)
    if dataset in SPECS:
        return locate_data(root) / dataset
    root = Path(root)
    for parent in (root, root / "data"):
        directory = parent / dataset
        if (directory / "native_manifest.json").is_file():
            return directory
    raise FileNotFoundError(
        f"Missing native_manifest.json for {dataset}; run scripts/convert_native.py"
    )


def native_reference(source, dataset):
    path = source / "native_manifest.json"
    native = json.loads(path.read_text())
    if (
        native["format"] != NATIVE_FORMAT
        or native["dataset"] != dataset
        or native["raw_units_per_day"] != SECONDS_PER_DAY
    ):
        raise ValueError("Unsupported or mismatched native dataset manifest")
    files = (
        "stat.txt",
        "entity2id.json",
        "relation2id.json",
        *(f"{s}.txt" for s in SPLITS),
    )
    hashes = {name: native["source_sha256"][name] for name in files}
    hashes["native_manifest.json"] = sha256(path)
    return native, hashes


def prepare_dataset(dataset, data_root, cache_root):
    source = dataset_directory(data_root, dataset)
    output = Path(cache_root) / dataset
    output.mkdir(parents=True, exist_ok=True)
    native, native_hashes = None, {}
    if dataset in SPECS:
        n, r, counts, units = SPECS[dataset]
    else:
        native, native_hashes = native_reference(source, dataset)
        for name, value in native_hashes.items():
            if sha256(source / name) != value:
                raise ValueError(f"Native input changed: {dataset}/{name}")
        n, r, units = (
            native["entities"],
            native["relations"],
            native["raw_units_per_day"],
        )
        counts = [native["splits"][split]["rows"] for split in SPLITS]
    stat = [int(x) for x in (source / "stat.txt").read_text().split()]
    if stat[:2] != [n, r] or min(n, r) < 1:
        raise ValueError("Vocabulary differs from the configured dataset")
    manifest = {
        "dataset": dataset,
        "entities": n,
        "relations": r,
        "raw_units_per_day": units,
        "source_sha256": native_hashes or {"stat.txt": sha256(source / "stat.txt")},
        "splits": {},
    }
    previous = None
    for split, expected in zip(SPLITS, counts):
        path = source / f"{split}.txt"
        array = np.loadtxt(path, dtype=np.int64, usecols=(0, 1, 2, 3), ndmin=2)
        if len(array) != expected or array.shape[1] != 4 or not len(array):
            raise ValueError(f"Unexpected {dataset}/{split} dimensions: {array.shape}")
        if (
            np.any(array[:, :3] < 0)
            or np.any(array[:, [0, 2]] >= n)
            or np.any(array[:, 1] >= r)
        ):
            raise ValueError("Entity/relation ID outside released vocabulary")
        if np.any(np.diff(array[:, 3]) < 0):
            raise ValueError("Released event rows are not chronological")
        times = np.unique(array[:, 3])
        if native is not None:
            train_end = native["train_until_exclusive_utc_seconds"]
            valid_end = native["valid_until_exclusive_utc_seconds"]
            if (
                (split == "train" and (len(times) < 2 or times[-1] >= train_end))
                or (
                    split == "valid"
                    and (times[0] < train_end or times[-1] >= valid_end)
                )
                or (split == "test" and times[0] < valid_end)
            ):
                raise ValueError(
                    f"Native {split} timestamps do not match the conversion cutoffs"
                )
        if previous is not None and times[0] <= previous:
            raise ValueError("Chronological splits overlap")
        previous = times[-1]
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=output, suffix=".npy", delete=False
            ) as stream:
                temporary = Path(stream.name)
                np.save(stream, array)
            os.replace(temporary, output / f"{split}.npy")
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        manifest["source_sha256"][f"{split}.txt"] = sha256(path)
        manifest["splits"][split] = {
            "rows": len(array),
            "timestamps": len(times),
            "raw_time_min": int(times[0]),
            "raw_time_max": int(times[-1]),
            "day_min": float(times[0] / units),
            "day_max": float(times[-1] / units),
            "array_sha256": sha256(output / f"{split}.npy"),
        }
    reference = ROOT / "splits" / f"{dataset}.json"
    if reference.exists():
        expected = json.loads(reference.read_text())
        if manifest != expected:
            raise ValueError(
                "Dataset fingerprints differ from bundled split identifiers"
            )
    dump_json(output / "manifest.json", manifest)
    return manifest


class TemporalDataset:
    def __init__(self, dataset, data_root=DATA_ROOT, cache_root=None):
        self.name = dataset
        source = dataset_directory(data_root, dataset)
        self.data_root = source.parent
        self.cache_root = Path(cache_root or ROOT / "cache")
        manifest_path = self.cache_root / dataset / "manifest.json"
        if not manifest_path.exists():
            prepare_dataset(dataset, self.data_root, self.cache_root)
        self.manifest = json.loads(manifest_path.read_text())
        if dataset in SPECS:
            reference = ROOT / "splits" / f"{dataset}.json"
            if self.manifest != json.loads(reference.read_text()):
                raise ValueError(
                    "Prepared data do not match the released split reference"
                )
        else:
            native, hashes = native_reference(source, dataset)
            if (
                self.manifest["dataset"] != dataset
                or self.manifest["source_sha256"] != hashes
                or any(
                    self.manifest[key] != native[key]
                    for key in ("entities", "relations", "raw_units_per_day")
                )
            ):
                raise ValueError(
                    "Prepared data do not match the native manifest. Use a fresh cache or rerun prepare_data.py."
                )
        for name, value in self.manifest["source_sha256"].items():
            if sha256(self.data_root / dataset / name) != value:
                raise ValueError(f"Released input changed: {dataset}/{name}")
        self.entities = self.manifest["entities"]
        self.relations = self.manifest["relations"]
        self.units = self.manifest["raw_units_per_day"]
        self.raw = {}
        for split in ["train", "valid", "test"]:
            path = self.cache_root / dataset / f"{split}.npy"
            if sha256(path) != self.manifest["splits"][split]["array_sha256"]:
                raise ValueError("Prepared array is corrupted; run prepare_data again")
            self.raw[split] = np.load(path, mmap_mode="r")
        self.origin = float(self.raw["train"][:, 3].min())
        self.event_times = {
            s: (a[:, 3].astype(np.float64) - self.origin) / self.units
            for s, a in self.raw.items()
        }
        self.fingerprint = fingerprint(self.manifest)
        self._index()

    def _index(self):
        self.rows = {}
        self.times = {}
        self.split_times = {}
        self.row_ids = {}
        for split, array in self.raw.items():
            order = np.argsort(self.event_times[split], kind="stable")
            timestamps = self.event_times[split][order]
            unique, starts = np.unique(timestamps, return_index=True)
            self.times[split] = timestamps
            self.row_ids[split] = order
            self.rows[split] = array[order, :3]
            self.split_times[split] = {
                float(t): self.rows[split][a:b]
                for t, a, b in zip(unique, starts, [*starts[1:], len(order)])
            }
        self.all_times = np.concatenate(list(self.times.values()))
        self.all_triples = np.concatenate(list(self.rows.values()))
        order = np.argsort(self.all_times, kind="stable")
        self.all_times = self.all_times[order]
        self.all_triples = self.all_triples[order]
        self.observation_times = np.unique(self.all_times)
        self.answer_cache = OrderedDict()

    def perturbed(self, times):
        clone = object.__new__(type(self))
        clone.__dict__ = dict(self.__dict__)
        if (
            times.shape != self.event_times["test"].shape
            or not np.isfinite(times).all()
        ):
            raise ValueError("Invalid perturbed test times")
        clone.event_times = {
            **self.event_times,
            "test": np.asarray(times, dtype=np.float64),
        }
        clone._index()
        return clone

    def answers(self, time):
        if time in self.answer_cache:
            self.answer_cache.move_to_end(time)
            return self.answer_cache[time]
        left = int(np.searchsorted(self.all_times, time, side="left"))
        right = int(np.searchsorted(self.all_times, time, side="right"))
        answers = defaultdict(set)
        for s, r, o in self.all_triples[int(left) : right]:
            answers[(int(s), int(r))].add(int(o))
        self.answer_cache[time] = answers
        while len(self.answer_cache) > 64:
            self.answer_cache.popitem(last=False)
        return answers


@dataclass
class ForecastHistory:
    initialization: EventGraph
    intervals: list
    first_boundary: float
    observed_through: float
    target_time: float


class HistoryBuilder:
    def __init__(self, data, device, history_days, graph_limit=0):
        self.data, self.device, self.history_days = (
            data,
            torch.device(device),
            history_days,
        )
        self.graph_limit = graph_limit
        self.graph_cache = OrderedDict()

    def _graph(self, key, rows):
        if key not in self.graph_cache:
            if self.graph_limit:
                rows = rows[: self.graph_limit]
            self.graph_cache[key] = EventGraph(
                rows, self.data.entities, self.data.relations, self.device
            )
            while len(self.graph_cache) > 64:
                self.graph_cache.popitem(last=False)
        self.graph_cache.move_to_end(key)
        return self.graph_cache[key]

    def build(self, target_time, observed_through=None, history_days=None):
        data = self.data
        cutoff = (
            float(np.nextafter(target_time, -np.inf))
            if observed_through is None
            else float(observed_through)
        )
        if cutoff >= target_time:
            raise ValueError(
                "Observed cutoff must strictly precede the prediction target"
            )
        if observed_through is not None and cutoff < data.observation_times[0]:
            raise ValueError(
                "No observed history exists at the requested open-loop cutoff"
            )
        width = self.history_days if history_days is None else history_days
        anchor = target_time if observed_through is None else cutoff
        first = max(float(data.observation_times[0]), anchor - width)
        if first >= target_time:
            first = target_time - min(width, 1 / data.units)
        initial_end = int(np.searchsorted(data.all_times, first, side="right"))
        visible_end = int(np.searchsorted(data.all_times, cutoff, side="right"))
        if initial_end > visible_end:
            initial_end = visible_end
        graph = self._graph(("prefix", initial_end), data.all_triples[:initial_end])
        initialization = graph
        observed = data.observation_times[
            (data.observation_times > first) & (data.observation_times <= cutoff)
        ]
        boundaries = [first, *observed.tolist(), target_time]
        intervals = []
        for i, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
            if i:
                left = int(
                    np.searchsorted(data.all_times, boundaries[i - 1], side="right")
                )
                right = int(np.searchsorted(data.all_times, start, side="right"))
                graph = self._graph(
                    ("interval", left, right), data.all_triples[left:right]
                )
            if end > start:
                intervals.append((start, end, graph))
        return ForecastHistory(
            initialization, intervals, first, cutoff, float(target_time)
        )

    def continuation(self, start, target):
        times = self.data.observation_times
        position = int(np.searchsorted(times, start, side="left"))
        if position >= len(times) or times[position] != start or target <= start:
            raise ValueError("Continuation must begin at an observed boundary")
        boundaries = [
            start,
            *times[(times > start) & (times < target)].tolist(),
            target,
        ]
        result = []
        for left, right in zip(boundaries, boundaries[1:]):
            a = int(np.searchsorted(self.data.all_times, left, side="left"))
            b = int(np.searchsorted(self.data.all_times, left, side="right"))
            graph = self._graph(("interval", a, b), self.data.all_triples[a:b])
            result.append((left, right, graph))
        return result


def main():
    parser = argparse.ArgumentParser(
        description="Validate released data and prepare immutable split caches"
    )
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--cache-root", type=Path, default=ROOT / "cache")
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    args = parser.parse_args()
    for dataset in args.datasets:
        manifest = prepare_dataset(dataset, args.data_root, args.cache_root)
        print(
            json.dumps(
                {
                    "dataset": dataset,
                    "rows": {k: v["rows"] for k, v in manifest["splits"].items()},
                    "raw_units_per_day": manifest["raw_units_per_day"],
                }
            ),
            flush=True,
        )
