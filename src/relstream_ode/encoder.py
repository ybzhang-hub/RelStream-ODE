"""Joint R-GCN initialization with simultaneous entity/relation updates."""

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


class JointLayer(nn.Module):
    def __init__(self, entity_dim, relation_dim, relations):
        super().__init__()
        self.typed_weights = nn.Parameter(
            torch.empty(relations, entity_dim, entity_dim)
        )
        self.self_entity = nn.Linear(entity_dim, entity_dim, bias=False)
        self.endpoint = nn.Linear(2 * entity_dim, relation_dim, bias=False)
        self.self_relation = nn.Linear(relation_dim, relation_dim, bias=False)
        self.relation_context = nn.Linear(relation_dim, relation_dim, bias=False)
        for matrix in self.typed_weights:
            nn.init.xavier_uniform_(matrix)
        for layer in [
            self.self_entity,
            self.endpoint,
            self.self_relation,
            self.relation_context,
        ]:
            nn.init.xavier_uniform_(layer.weight)

    def forward(self, entities, relations, graph):
        incoming = torch.zeros_like(entities)
        for group in graph.groups:
            transformed = F.linear(
                entities[group.unique_sources], self.typed_weights[group.relation]
            )
            messages = (
                transformed[group.source_inverse]
                * group.incoming_normalization[:, None]
            )
            incoming = incoming.index_add(0, group.destination, messages)
        # The endpoint map is linear, so averaging endpoints first is exact.
        starts = torch.sparse.mm(graph.source_mean, entities)
        ends = torch.sparse.mm(graph.destination_mean, entities)
        context = self.endpoint(torch.cat([starts, ends], dim=-1))
        return F.silu(self.self_entity(entities) + incoming), F.silu(
            self.self_relation(relations) + self.relation_context(context)
        )


class JointInitializer(nn.Module):
    def __init__(self, entities, relations, config):
        super().__init__()
        self.entity = nn.Parameter(torch.empty(entities, config["entity_dim"]))
        self.relation = nn.Parameter(torch.empty(relations, config["relation_dim"]))
        nn.init.xavier_uniform_(self.entity)
        nn.init.xavier_uniform_(self.relation)
        self.layers = nn.ModuleList(
            [
                JointLayer(config["entity_dim"], config["relation_dim"], relations)
                for _ in range(config["initializer_layers"])
            ]
        )

    def forward(self, graph, checkpoint_layers=False):
        h, r = self.entity, self.relation
        for layer in self.layers:
            if checkpoint_layers and torch.is_grad_enabled():
                h, r = checkpoint(
                    layer, h, r, graph, use_reentrant=False, preserve_rng_state=False
                )
            else:
                h, r = layer(h, r, graph)
        return h, r
