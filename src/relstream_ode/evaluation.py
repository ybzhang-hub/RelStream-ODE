"""Full-entity directional ranking with time-aware truth filtering."""

import numpy as np
import torch

from .solver import integrate


def filtered_ranks(scores, queries, answers, tie="average"):
    if scores.ndim != 2 or not torch.isfinite(scores).all():
        raise FloatingPointError("Expected finite full-vocabulary scores")
    target = scores.gather(1, queries[:, 2:3])
    filtered = scores.clone()
    rows, columns = [], []
    for i, (s, r, o) in enumerate(queries.detach().cpu().tolist()):
        other = sorted(answers.get((s, r), set()) - {o})
        rows.extend([i] * len(other))
        columns.extend(other)
    if rows:
        filtered[
            torch.as_tensor(rows, device=scores.device),
            torch.as_tensor(columns, device=scores.device),
        ] = -torch.inf
    higher = (filtered > target).sum(1)
    equal = (filtered == target).sum(1)
    if tie == "average":
        return 1 + higher.double() + 0.5 * (equal.double() - 1)
    if tie == "optimistic":
        return 1 + higher.double()
    if tie == "pessimistic":
        return higher.double() + equal.double()
    raise ValueError("Unsupported tie handling")


def summarize_ranks(ranks):
    ranks = np.asarray(ranks, dtype=np.float64)
    if not len(ranks) or not np.isfinite(ranks).all():
        raise ValueError("No finite evaluation ranks")
    return {
        "MRR": float(np.mean(1 / ranks)),
        **{f"Hits@{k}": float(np.mean(ranks <= k)) for k in [1, 3, 10]},
        "queries": len(ranks),
    }


def evaluate(
    model,
    data,
    builder,
    split="valid",
    batch_size=32,
    tie="average",
    max_times=0,
    max_queries=0,
    collector=None,
    progress=None,
    offset_days=None,
    shift_query_days=None,
    history_days_override=None,
    selected_times=None,
    reuse_state=True,
):
    was_training = model.training
    model.eval()
    ranks = []
    ids, triples, query_times, label_times = [], [], [], []
    times = (
        list(data.split_times[split])
        if selected_times is None
        else list(selected_times)
    )[: max_times or None]
    cached_state, cached_time, cached_first = None, None, None
    if offset_days is not None and shift_query_days is not None:
        raise ValueError(
            "Choose lead-time evaluation or fixed-history query-time shifts"
        )
    try:
        # Graph-index tensors remain reusable by subsequent training epochs.
        with torch.no_grad():
            for index, time in enumerate(times):
                rows = data.split_times[split][time][: max_queries or None]
                cutoff = None if offset_days is None else time - offset_days
                target = time
                if shift_query_days is not None:
                    previous = (
                        int(np.searchsorted(data.observation_times, time, side="left"))
                        - 1
                    )
                    if previous < 0:
                        raise ValueError("No observed history precedes this query")
                    cutoff = float(data.observation_times[previous])
                    target = cutoff + shift_query_days
                width = (
                    (100.0 if cutoff is not None else builder.history_days)
                    if history_days_override is None
                    else history_days_override
                )
                first = max(
                    float(data.observation_times[0]),
                    (target if cutoff is None else cutoff) - width,
                )
                can_reuse = (
                    reuse_state
                    and cutoff is None
                    and cached_state is not None
                    and first == cached_first
                    and time > cached_time
                )
                if can_reuse:
                    raw_state = integrate(
                        model.field,
                        cached_state,
                        builder.continuation(cached_time, time),
                    )
                    if not all(torch.isfinite(value).all() for value in raw_state):
                        raise FloatingPointError("Nonfinite continued evaluation state")
                else:
                    history = builder.build(
                        target, observed_through=cutoff, history_days=width
                    )
                    raw_state = model.evolve(history)
                cached_state, cached_time, cached_first = raw_state, time, first
                state = model.decoder.prepare(*raw_state)
                answers = data.answers(time)
                row_start = int(np.searchsorted(data.times[split], time, side="left"))
                for begin in range(0, len(rows), batch_size):
                    batch = rows[begin : begin + batch_size]
                    queries = torch.as_tensor(batch.copy(), device=builder.device)
                    scores = model.decoder(*state, queries[:, 0], queries[:, 1])
                    rank = filtered_ranks(scores, queries, answers, tie).cpu().numpy()
                    ranks.extend(rank.tolist())
                    if collector is not None:
                        triples.extend(batch.tolist())
                        query_times.extend([target] * len(batch))
                        label_times.extend([time] * len(batch))
                        ids.extend(
                            data.row_ids[split][
                                row_start + begin : row_start + begin + len(batch)
                            ].tolist()
                        )
                if progress is not None:
                    progress(
                        {
                            "split": split,
                            "completed_timestamps": index + 1,
                            "timestamps": len(times),
                            "queries": len(ranks),
                        }
                    )
    finally:
        model.train(was_training)
    if collector is not None:
        collector.update(
            ranks=np.asarray(ranks),
            queries=np.asarray(triples, dtype=np.int64),
            days=np.asarray(query_times, dtype=np.float64),
            row_ids=np.asarray(ids, dtype=np.int64),
            original_label_days=np.asarray(label_times, dtype=np.float64),
        )
    return summarize_ranks(ranks)
