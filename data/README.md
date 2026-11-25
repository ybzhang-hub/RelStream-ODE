# Data layout

Use the processed splits distributed with [RE-GCN](https://github.com/Lee-zix/RE-GCN). Keep their entity and relation IDs and chronological split membership.

```text
<data-root>/
├── ICEWS05-15/
├── ICEWS14/
├── ICEWS18/
└── GDELT/
    ├── stat.txt
    ├── train.txt
    ├── valid.txt
    └── test.txt
```

Each fact starts with `subject relation object timestamp`; any fifth column is ignored. `stat.txt` starts with the entity and relation counts. Static entity-word graphs and reciprocal edges are not used.

Pass `--data-root <data-root>` or set `RELSTREAM_DATA_ROOT`. The parent RE-GCN directory is also accepted. Preparation checks source hashes against `splits/*.json`, builds array caches outside the raw data, and preserves all target rows.

Raw time is measured in days for ICEWS05-15, hours for the supplied ICEWS14/18, and minutes for GDELT. The respective divisors are 1, 24, 24 and 1440. The alternative ICEWS14s_before_reformat directory is not the configured ICEWS14 input. GDELT has 15-minute observation bins and spans about 31 actual days; snapshot offsets and day offsets are different units.

Data are supplied separately and remain subject to their original providers' terms.

## Native exports

The native converter accepts multiple event files from a single provider. Inputs can be plain tab-delimited text, gzip-compressed text, or ZIPs containing one event table. It reads local files supplied by the user.

| Provider | Input | Ordered endpoints | Relation | Time |
| --- | --- | --- | --- | --- |
| ICEWS | Headered Coded Event Data table | `Source Name`, `Target Name` | `CAMEO Code` | `Event Date`, represented at UTC midnight at its native one-day resolution |
| GDELT 2.0 | Headerless 61-column Events export, including tab-delimited `.export.CSV` | `Actor1Code`, `Actor2Code` | `EventCode` | `DATEADDED`, UTC database availability time at the feed's native resolution |

GDELT's event date field gives the day of occurrence and its availability field gives the database arrival time. See the [GDELT 2.0 event codebook](https://data.gdeltproject.org/documentation/GDELT-Event_Codebook-V2.0.pdf) and the [ICEWS catalog](https://doi.org/10.7910/DVN/28075).

The converter trims surrounding whitespace and preserves CAMEO codes as strings, including leading zeroes. It excludes rows with missing endpoints, codes or timestamps, rejects malformed table schemas, and counts invalid dates, codes and duplicate `(subject, relation, object, timestamp)` rows. It keeps edge direction and retains gaps and tied timestamps.

`--train-until` and `--valid-until` are exclusive endpoints of their respective splits. Events exactly at a cutoff start the next split. Actor and relation IDs come from the lexicographically sorted values in training. Validation and test rows with actors or relations absent from that vocabulary are excluded and counted per split as `excluded_unseen`. Candidate ranking uses the entire training vocabulary. A one-row embedding table uses stored BatchNorm statistics because a single row cannot estimate batch variance.

The dataset directory contains `train.txt`, `valid.txt`, `test.txt`, `stat.txt`, two JSON ID maps, and `native_manifest.json`. The fourth column stores integer UTC seconds. RelStream-ODE converts these to elapsed days at load time and forecasts from strictly earlier observations. The manifest records both cutoff seconds, the time field, input and output checksums, and the retained and excluded counts. The benchmark split manifests remain the source of truth for the four released benchmarks.
