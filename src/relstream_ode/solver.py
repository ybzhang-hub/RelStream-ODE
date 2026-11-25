"""Three classical RK4 substeps per graph interval with direct gradients."""

from functools import partial

import torch
from torch.utils.checkpoint import checkpoint


def _add(state, delta, scale):
    return tuple(x + scale * dx for x, dx in zip(state, delta))


def rk4_interval(field, state, start, end, graph):
    if end <= start:
        raise ValueError("Integration intervals must have positive duration")
    step = (end - start) / 3
    for i in range(3):
        t = start + i * step
        k1 = field(t, state, graph)
        k2 = field(t + step / 2, _add(state, k1, step / 2), graph)
        k3 = field(t + step / 2, _add(state, k2, step / 2), graph)
        k4 = field(t + step, _add(state, k3, step), graph)
        state = tuple(
            x + (step / 6) * (a + 2 * b + 2 * c + d)
            for x, a, b, c, d in zip(state, k1, k2, k3, k4)
        )
    return state


def _segment(h, r, field, intervals):
    state = (h, r)
    for start, end, graph in intervals:
        state = rk4_interval(field, state, start, end, graph)
    return state


def integrate(field, state, intervals, checkpoint_segments=0):
    size = checkpoint_segments or len(intervals) or 1
    for begin in range(0, len(intervals), size):
        segment = intervals[begin : begin + size]
        function = partial(_segment, field=field, intervals=segment)
        if checkpoint_segments and torch.is_grad_enabled():
            state = checkpoint(
                function, *state, use_reentrant=False, preserve_rng_state=False
            )
        else:
            state = function(*state)
    return state
