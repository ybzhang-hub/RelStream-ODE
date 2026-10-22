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
    offset_steps=None,
    shift_query_steps=None,
    history_steps_override=None,
    selected_times=None,
    reuse_state=True,
):
    """Rank immutable ground-truth queries under strictly original event availability.

    Perturbed datasets expose per-query *time-code offsets* only. In this
    diagnostic, graph events, history eligibility, integration boundaries,
    true targets, and filtered answers all keep original timestamps. A noisy
    query re-evolves its original history with the shifted time-code input.
    This is a time-feature stress test, NOT moving event arrival times.
    """
    if offset_steps is not None and shift_query_steps is not None:
        raise ValueError("Choose lead-time evaluation or fixed-label clock shift")
    was_training = model.training
    previous_shift = model.field.time_code_shift
    model.eval()
    ranks = []
    ids, triples, query_times, label_times, input_shifts = [], [], [], [], []
    times = (
        list(data.split_times[split]) if selected_times is None else list(selected_times)
    )[: max_times or None]
    cached_state, cached_time, cached_first = None, None, None
    try:
        with torch.no_grad():
            for index, time in enumerate(times):
                rows = data.split_times[split][time][: max_queries or None]
                cutoff = None if offset_steps is None else float(time - offset_steps)
                target = float(time)
                if shift_query_steps is not None:
                    previous = int(np.searchsorted(
                        data.observation_times, time, side="left"
                    )) - 1
                    if previous < 0:
                        raise ValueError("No observed history precedes this query")
                    cutoff = float(data.observation_times[previous])
                    target = cutoff + shift_query_steps
                width = (
                    builder.history_window
                    if history_steps_override is None else history_steps_override
                )
                first = max(
                    float(data.observation_times[0]),
                    (target if cutoff is None else cutoff) - width,
                )
                row_start = int(np.searchsorted(data.times[split], time, side="left"))
                original_ids = data.row_ids[split][row_start:row_start + len(rows)]
                offsets = getattr(data, "query_time_offsets", None)
                if offsets is None or split != "test":
                    row_shifts = np.zeros(len(rows), dtype=np.float64)
                else:
                    row_shifts = offsets[original_ids]
                if not np.isfinite(row_shifts).all():
                    raise ValueError("All query time-code perturbations must be finite")
                # Stable partition retains every target, including same-timestamp
                # truths with different perturbations; answer sets never change.
                unique_offsets, inverse = np.unique(row_shifts, return_inverse=True)
                truth = data.answers(time)
                for group_index, shift in enumerate(unique_offsets):
                    member_indices = np.flatnonzero(inverse == group_index)
                    local_rows = rows[member_indices]
                    model.field.time_code_shift = float(shift)
                    can_reuse = (
                        reuse_state
                        and shift == 0
                        and cutoff is None
                        and cached_state is not None
                        and first == cached_first
                        and time > cached_time
                    )
                    if can_reuse:
                        raw_state = integrate(
                            model.field, cached_state,
                            builder.continuation(cached_time, time),
                            substeps=model.config["rk4_substeps"],
                        )
                        if not all(torch.isfinite(x).all() for x in raw_state):
                            raise FloatingPointError("Nonfinite continued state")
                    else:
                        history = builder.build(
                            target, observed_through=cutoff, history_window=width
                        )
                        raw_state = model.evolve(history)
                    if shift == 0 and cutoff is None:
                        cached_state, cached_time, cached_first = raw_state, time, first
                    else:
                        cached_state, cached_time, cached_first = None, None, None
                    state = model.decoder.prepare(*raw_state)
                    for begin in range(0, len(local_rows), batch_size):
                        members = member_indices[begin:begin + batch_size]
                        batch = local_rows[begin:begin + batch_size]
                        queries = torch.as_tensor(batch.copy(), device=builder.device)
                        scores = model.decoder(*state, queries[:, 0], queries[:, 1])
                        rank = filtered_ranks(scores, queries, truth, tie).cpu().numpy()
                        ranks.extend(rank.tolist())
                        if collector is not None:
                            triples.extend(batch.tolist())
                            query_times.extend([target] * len(batch))
                            label_times.extend([time] * len(batch))
                            input_shifts.extend([float(shift)] * len(batch))
                            ids.extend(original_ids[members].tolist())
                if progress is not None:
                    progress({
                        "split": split,
                        "completed_timestamps": index + 1,
                        "timestamps": len(times),
                        "queries": len(ranks),
                    })
    finally:
        model.field.time_code_shift = previous_shift
        model.train(was_training)
    if collector is not None:
        collector.update(
            ranks=np.asarray(ranks),
            queries=np.asarray(triples, dtype=np.int64),
            days=np.asarray(query_times, dtype=np.float64),
            row_ids=np.asarray(ids, dtype=np.int64),
            original_label_days=np.asarray(label_times, dtype=np.float64),
            time_code_offsets=np.asarray(input_shifts, dtype=np.float64),
        )
    return summarize_ranks(ranks)
