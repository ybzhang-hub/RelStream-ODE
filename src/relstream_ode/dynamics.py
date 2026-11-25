"""Ordered-pair relation dynamics and dense low-rank graph transport."""

import math

import torch
from torch import nn
from torch.nn import functional as F


class JointVectorField(nn.Module):
    def __init__(self, relations, config):
        super().__init__()
        d, dr, dt, p = (
            config[k]
            for k in ["entity_dim", "relation_dim", "time_dim", "transport_rank"]
        )
        self.variant = config["variant"]
        self.epsilon = config["layer_norm_epsilon"]
        self.pair_encoder = nn.Sequential(
            nn.Linear(2 * d, config["pair_hidden_dim"]),
            nn.SiLU(),
            nn.Linear(config["pair_hidden_dim"], d),
        )
        self.relation_flow = nn.Sequential(
            nn.Linear(dr + d + dt, config["relation_hidden_dim"]),
            nn.SiLU(),
            nn.Linear(config["relation_hidden_dim"], dr),
        )
        self.transport_coefficients = nn.Sequential(
            nn.Linear(dr, config["transport_hidden_dim"]),
            nn.SiLU(),
            nn.Linear(config["transport_hidden_dim"], p),
            nn.Tanh(),
        )
        self.left = nn.Parameter(torch.empty(relations, d, p))
        self.right = nn.Parameter(torch.empty(relations, d, p))
        for factor in [self.left, self.right]:
            for matrix in factor:
                nn.init.xavier_uniform_(matrix)
        self.query = nn.Linear(d, d // 8, bias=False)
        self.key = nn.Linear(d, d // 8, bias=False)
        self.gate = nn.Linear(2 * d + dr, d)
        self.value = nn.Linear(d, d, bias=False)
        self.entity_flow = nn.Linear(d, d, bias=False)
        frequencies = torch.exp(
            -math.log(config["time_frequency_base"])
            * torch.arange(0, dt, 2, dtype=torch.float32)
            / dt
        )
        self.register_buffer("frequencies", frequencies)
        self.nfe = 0
        if self.variant == "no_relation_evolution":
            for module in [self.pair_encoder, self.relation_flow]:
                module.requires_grad_(False)
        if self.variant in {"identity_transport", "no_graph_flow"}:
            self.transport_coefficients.requires_grad_(False)
            self.left.requires_grad_(False)
            self.right.requires_grad_(False)
        if self.variant == "no_graph_flow":
            for module in [
                self.query,
                self.key,
                self.gate,
                self.value,
                self.entity_flow,
            ]:
                module.requires_grad_(False)

    def time_code(self, time):
        angles = (
            torch.as_tensor(
                time, device=self.frequencies.device, dtype=self.frequencies.dtype
            )
            * self.frequencies
        )
        return torch.stack([angles.sin(), angles.cos()], dim=-1).flatten()

    def forward(self, time, state, graph):
        self.nfe += 1
        h, r = state
        if self.variant == "no_relation_evolution":
            dr = torch.zeros_like(r)
        else:
            usage = h.new_zeros((len(r), h.shape[1]))
            if graph.num_edges:
                pairs = self.pair_encoder(
                    torch.cat([h[graph.pairs[:, 0]], h[graph.pairs[:, 1]]], dim=-1)
                )
                usage = (
                    usage.index_add(0, graph.relation, pairs[graph.pair_inverse])
                    / graph.relation_counts[:, None]
                )
            code = self.time_code(time).expand(len(r), -1)
            dr = self.relation_flow(torch.cat([r, usage, code], dim=-1))
        dh = torch.zeros_like(h)
        if self.variant == "no_graph_flow" or not graph.num_edges:
            return dh, dr
        operators = None
        if self.variant != "identity_transport":
            active = graph.active_relations
            coefficients = self.transport_coefficients(r[active])
            operators = torch.bmm(
                self.left[active] * coefficients[:, None, :],
                self.right[active].transpose(1, 2),
            )
        messages = torch.zeros_like(h)
        for slot, group in enumerate(graph.groups):
            source, destination = h[group.source], h[group.destination]
            difference = destination - source
            transported = (
                difference
                if operators is None
                else F.linear(difference, operators[slot])
            )
            compatibility = (self.query(source) * self.key(destination)).sum(
                -1
            ) / math.sqrt(self.query.out_features)
            relation = r[group.relation].expand(len(source), -1)
            gate = torch.sigmoid(
                self.gate(torch.cat([source, destination, relation], dim=-1))
            )
            edge_messages = (
                compatibility[:, None] * gate * torch.tanh(self.value(transported))
            )
            messages = messages.index_add(0, group.source, edge_messages)
        active_messages = self.entity_flow(messages[graph.active_sources])
        derivative = F.layer_norm(
            active_messages, (h.shape[1],), weight=None, bias=None, eps=self.epsilon
        )
        return dh.index_copy(0, graph.active_sources, derivative), dr
