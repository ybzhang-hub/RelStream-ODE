"""Joint initialization, causal interval evolution and target-time decoding."""

import torch
from torch import nn

from .decoder import TuckerDecoder
from .dynamics import JointVectorField
from .encoder import JointInitializer
from .solver import integrate


class RelStreamODE(nn.Module):
    def __init__(self, entities, relations, config):
        super().__init__()
        self.config = dict(config)
        self.initializer = JointInitializer(entities, relations, config)
        self.field = JointVectorField(relations, config)
        self.decoder = TuckerDecoder(config)

    def evolve(self, history, checkpoint_segments=0):
        state = self.initializer(
            history.initialization, checkpoint_layers=bool(checkpoint_segments)
        )
        state = integrate(self.field, state, history.intervals, checkpoint_segments)
        if not all(torch.isfinite(x).all() for x in state):
            raise FloatingPointError(
                "Nonfinite evolved state; inspect times, interval lengths and optimizer diagnostics"
            )
        return state

    def forward(self, history, queries, checkpoint_segments=0):
        entities, relations = self.decoder.prepare(
            *self.evolve(history, checkpoint_segments)
        )
        return self.decoder(entities, relations, queries[:, 0], queries[:, 1])
