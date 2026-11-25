"""Convert native event exports to chronological, trainable directed streams."""

import argparse
import csv
import gzip
import io
import re
import sqlite3
import tempfile
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile

from .config import DATASETS
from .utils import dump_json, sha256

NATIVE_FORMAT = "relstream-native-v1"
SECONDS_PER_DAY = 86400
SPLITS = ("train", "valid", "test")
ICEWS_COLUMNS = ("Event Date", "Source Name", "CAMEO Code", "Target Name")


def validate_dataset_name(name):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name):
        raise ValueError(
            "Dataset names must contain only letters, digits, hyphens and underscores"
        )


@contextmanager
def open_export(path):
    if path.suffix.lower() == ".zip":
        with ZipFile(path) as archive:
            members = [
                i
                for i in archive.infolist()
                if not i.is_dir()
                and Path(i.filename).suffix.lower() in {".csv", ".tsv", ".tab", ".txt"}
            ]
            if len(members) != 1:
                raise ValueError(f"{path.name}: expected one event table in the ZIP")
            with (
                archive.open(members[0]) as stream,
                io.TextIOWrapper(stream, encoding="utf-8-sig", newline="") as text,
            ):
                yield text
    else:
        opener = gzip.open if path.suffix.lower() == ".gz" else open
        with opener(path, "rt", encoding="utf-8-sig", newline="") as text:
            yield text


def export_rows(path, provider):
    with open_export(path) as stream:
        if provider == "icews":
            reader = csv.reader(stream, delimiter="\t", strict=True)
            header = [name.strip() for name in next(reader, [])]
            if any(header.count(name) != 1 for name in ICEWS_COLUMNS):
                raise ValueError(
                    f"{path.name}: expected ICEWS columns {', '.join(ICEWS_COLUMNS)}"
                )
            date, source, relation, target = [
                header.index(name) for name in ICEWS_COLUMNS
            ]
            width = len(header)
        else:
            reader = csv.reader(stream, delimiter="\t", quoting=csv.QUOTE_NONE)
            # GDELT 2.0 Events export positions, with zero-based indexing.
            date, source, relation, target = 59, 5, 26, 15
            width = 61
        for row in reader:
            if not row:
                continue
            if len(row) != width:
                raise ValueError(
                    f"{path.name}:{reader.line_num}: expected {width} columns, got {len(row)}"
                )
            yield tuple(row[i].strip() for i in (date, source, relation, target))


def utc_seconds(timestamp):
    value = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    if value.microsecond:
        raise ValueError("Use cutoffs with whole-second precision")
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return int(value.timestamp())


def ingest(database, inputs, provider):
    database.execute(
        "CREATE TABLE events (t INTEGER, s TEXT, r TEXT, o TEXT, PRIMARY KEY(t,s,r,o)) WITHOUT ROWID"
    )
    audit = Counter(
        input_rows=0,
        missing_fields=0,
        invalid_dates=0,
        invalid_codes=0,
        duplicate_quads=0,
    )
    date_format = "%Y-%m-%d" if provider == "icews" else "%Y%m%d%H%M%S"
    date_pattern = r"\d{4}-\d{2}-\d{2}" if provider == "icews" else r"\d{14}"

    def events(path):
        for date, source, relation, target in export_rows(path, provider):
            audit["input_rows"] += 1
            if not all((date, source, relation, target)):
                audit["missing_fields"] += 1
                continue
            if not re.fullmatch(r"[0-9]{2,4}", relation):
                audit["invalid_codes"] += 1
                continue
            try:
                if not re.fullmatch(date_pattern, date):
                    raise ValueError("Unexpected date syntax")
                timestamp = int(
                    datetime.strptime(date, date_format)
                    .replace(tzinfo=timezone.utc)
                    .timestamp()
                )
            except ValueError:
                audit["invalid_dates"] += 1
                continue
            yield timestamp, source, relation, target

    with database:
        for path in inputs:
            database.executemany(
                "INSERT OR IGNORE INTO events VALUES (?, ?, ?, ?)", events(path)
            )
    accepted = audit["input_rows"] - sum(
        audit[key] for key in ("missing_fields", "invalid_dates", "invalid_codes")
    )
    audit["duplicate_quads"] = accepted - database.total_changes
    return dict(audit)


def training_vocab(database, train_end):
    entities = set()
    relations = set()
    for source, relation, target in database.execute(
        "SELECT s,r,o FROM events WHERE t < ?", (train_end,)
    ):
        entities.update((source, target))
        relations.add(relation)
    return (
        {value: index for index, value in enumerate(sorted(entities))},
        {value: index for index, value in enumerate(sorted(relations))},
    )


def write_split(path, rows, entities, relations):
    counts = {"rows": 0, "excluded_unseen": 0, "timestamps": 0}
    previous = None
    with path.open("w", encoding="utf-8") as stream:
        for timestamp, source, relation, target in rows:
            if (
                source not in entities
                or target not in entities
                or relation not in relations
            ):
                counts["excluded_unseen"] += 1
                continue
            stream.write(
                f"{entities[source]}\t{relations[relation]}\t{entities[target]}\t{timestamp}\n"
            )
            counts["rows"] += 1
            if timestamp != previous:
                counts["timestamps"] += 1
                previous = timestamp
    return counts


def convert(provider, inputs, output, train_until, valid_until):
    output = Path(output)
    validate_dataset_name(output.name)
    if output.name in DATASETS:
        raise ValueError("Choose a unique native dataset name such as ICEWS-native")
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    train_end, valid_end = map(utc_seconds, (train_until, valid_until))
    if train_end >= valid_end:
        raise ValueError("The train cutoff must precede the validation cutoff")
    if provider not in {"icews", "gdelt2"} or not inputs:
        raise ValueError("Supply an ICEWS or GDELT 2.0 event export")
    inputs = [Path(path) for path in inputs]
    source_hashes = [{"file": path.name, "sha256": sha256(path)} for path in inputs]
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output.name}-", dir=output.parent
    ) as temporary:
        stage = Path(temporary) / "dataset"
        stage.mkdir()
        database = sqlite3.connect(Path(temporary) / "events.sqlite")
        try:
            audit = ingest(database, inputs, provider)
            entities, relations = training_vocab(database, train_end)
            split_counts = {}
            for split, condition, bounds in zip(
                SPLITS,
                ("t < ?", "t >= ? AND t < ?", "t >= ?"),
                ((train_end,), (train_end, valid_end), (valid_end,)),
            ):
                rows = database.execute(
                    f"SELECT t,s,r,o FROM events WHERE {condition} ORDER BY t,s,r,o",
                    bounds,
                )
                split_counts[split] = write_split(
                    stage / f"{split}.txt", rows, entities, relations
                )
        finally:
            database.close()
        if split_counts["train"]["timestamps"] < 2 or any(
            not counts["rows"] for counts in split_counts.values()
        ):
            raise ValueError(
                "Need at least two observed training timestamps and nonempty validation and test after vocabulary filtering"
            )
        (stage / "stat.txt").write_text(f"{len(entities)}\t{len(relations)}\n")
        dump_json(stage / "entity2id.json", entities)
        dump_json(stage / "relation2id.json", relations)
        files = [
            "stat.txt",
            "entity2id.json",
            "relation2id.json",
            *(f"{split}.txt" for split in SPLITS),
        ]
        manifest = {
            "format": NATIVE_FORMAT,
            "dataset": output.name,
            "provider": provider,
            "clock": "event_date_at_midnight_UTC"
            if provider == "icews"
            else "DATEADDED_UTC",
            "raw_units_per_day": SECONDS_PER_DAY,
            "entities": len(entities),
            "relations": len(relations),
            "vocabulary": "sorted_unique_training_actors_and_CAMEO_codes",
            "train_until_exclusive_utc_seconds": train_end,
            "valid_until_exclusive_utc_seconds": valid_end,
            "inputs": source_hashes,
            "conversion": audit,
            "splits": split_counts,
            "source_sha256": {name: sha256(stage / name) for name in files},
        }
        dump_json(stage / "native_manifest.json", manifest)
        stage.rename(output)
    return manifest


def main():
    parser = argparse.ArgumentParser(
        description="Convert provider event files into RelStream-ODE data"
    )
    parser.add_argument("--provider", required=True, choices=["icews", "gdelt2"])
    parser.add_argument("--inputs", required=True, nargs="+", type=Path)
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="New dataset directory; its basename is the dataset name",
    )
    parser.add_argument(
        "--train-until", required=True, help="Exclusive end of training (ISO 8601 UTC)"
    )
    parser.add_argument(
        "--valid-until",
        required=True,
        help="Exclusive end of validation (ISO 8601 UTC)",
    )
    args = parser.parse_args()
    report = convert(
        args.provider, args.inputs, args.output, args.train_until, args.valid_until
    )
    print(
        f"{report['dataset']}: "
        + ", ".join(
            f"{split}={counts['rows']}" for split, counts in report["splits"].items()
        )
    )
