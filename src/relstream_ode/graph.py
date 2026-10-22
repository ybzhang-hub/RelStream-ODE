"""Directed typed edge sets and parameter-independent graph indices."""

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class RelationEdges:
    relation: int
    source: torch.Tensor
    destination: torch.Tensor
    unique_sources: torch.Tensor
    source_inverse: torch.Tensor
    incoming_normalization: torch.Tensor


class EventGraph:
    def __init__(self, edges, entities, relations, device):
        edges = np.unique(np.asarray(edges, dtype=np.int64).reshape(-1, 3), axis=0)
        self.num_edges = len(edges)
        self.device = torch.device(device)
        self.groups = []
        self.source = torch.as_tensor(edges[:, 0].copy(), device=device)
        self.relation = torch.as_tensor(edges[:, 1].copy(), device=device)
        self.destination = torch.as_tensor(edges[:, 2].copy(), device=device)
        unique_pairs, pair_inverse = np.unique(
            edges[:, [0, 2]], axis=0, return_inverse=True
        )
        self.pairs = torch.as_tensor(unique_pairs.copy(), device=device)
        self.pair_inverse = torch.as_tensor(pair_inverse, device=device)
        self.active_sources = torch.unique(self.source, sorted=True)
        self.active_relations = torch.unique(self.relation, sorted=True)
        counts = np.bincount(edges[:, 1], minlength=relations)
        self.relation_counts = torch.as_tensor(
            counts.clip(min=1), dtype=torch.float32, device=device
        )
        for r in np.unique(edges[:, 1]):
            selected = edges[edges[:, 1] == r]
            src, inverse = np.unique(selected[:, 0], return_inverse=True)
            _, group, degree = np.unique(
                selected[:, 2], return_inverse=True, return_counts=True
            )
            self.groups.append(
                RelationEdges(
                    int(r),
                    torch.as_tensor(selected[:, 0].copy(), device=device),
                    torch.as_tensor(selected[:, 2].copy(), device=device),
                    torch.as_tensor(src, device=device),
                    torch.as_tensor(inverse, device=device),
                    torch.as_tensor(
                        (1.0 / degree[group]).astype(np.float32), device=device
                    ),
                )
            )
        values = (
            np.reciprocal(counts[edges[:, 1]].astype(np.float32))
            if len(edges)
            else np.empty(0, dtype=np.float32)
        )
        self.source_mean = self._incidence(
            edges[:, 1], edges[:, 0], values, relations, entities, device
        )
        self.destination_mean = self._incidence(
            edges[:, 1], edges[:, 2], values, relations, entities, device
        )

    @staticmethod
    def _incidence(relations, entities, values, n_relations, n_entities, device):
        indices = torch.as_tensor(
            np.stack([relations, entities]), dtype=torch.long, device=device
        )
        weights = torch.as_tensor(values, dtype=torch.float32, device=device)
        return torch.sparse_coo_tensor(
            indices, weights, (n_relations, n_entities), device=device
        ).coalesce()
