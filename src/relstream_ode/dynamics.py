"""Ordered-pair relation dynamics and configurable low-rank message transport."""

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
        self.transport_formula = config["transport_formula"]
        self.transport_implementation = config["transport_implementation"]
        self.usage_aggregation = (
            "max" if self.variant == "max_usage" else
            "attention" if self.variant == "attention_usage" else
            config["usage_aggregation"]
        )
        # Evaluation-only clock-input perturbation. Does not change event availability.
        self.time_code_shift = 0.0
        self.pair_encoder = nn.Sequential(
            nn.Linear(2 * d, config["pair_hidden_dim"]),
            nn.SiLU(),
            nn.Linear(config["pair_hidden_dim"], d),
        )
        self.usage_attention = (
            nn.Linear(d + dr, 1, bias=False)
            if self.usage_aggregation == "attention" else None
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
        self.static_coefficients = (
            nn.Parameter(torch.empty(relations, p).normal_(0, 0.1))
            if self.variant == "static_transport_coefficients" else None
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
        if self.variant in {"no_relation_evolution", "joint_ablation"}:
            for module in [self.pair_encoder, self.relation_flow]:
                module.requires_grad_(False)
        if self.variant in {"identity_transport", "no_graph_flow", "joint_ablation"}:
            self.transport_coefficients.requires_grad_(False)
            self.left.requires_grad_(False)
            self.right.requires_grad_(False)
        if self.variant in {"static_transport_coefficients"}:
            self.transport_coefficients.requires_grad_(False)
        if self.variant == "no_usage_summary":
            self.pair_encoder.requires_grad_(False)
        if self.variant == "no_vector_gate":
            self.gate.requires_grad_(False)
        if self.variant == "no_scalar_compatibility":
            self.query.requires_grad_(False)
            self.key.requires_grad_(False)
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
                time + self.time_code_shift,
                device=self.frequencies.device,
                dtype=self.frequencies.dtype
            )
            * self.frequencies
        )
        return torch.stack([angles.sin(), angles.cos()], dim=-1).flatten()

    def _usage_summary(self, h, r, graph):
        usage = h.new_zeros((len(r), h.shape[1]))
        if not graph.num_edges or self.variant == "no_usage_summary":
            return usage
        pairs = self.pair_encoder(
            torch.cat([h[graph.pairs[:, 0]], h[graph.pairs[:, 1]]], dim=-1)
        )
        encoded = pairs[graph.pair_inverse]
        if self.usage_aggregation == "mean":
            return usage.index_add(0, graph.relation, encoded) / graph.relation_counts[:, None]
        for relation in graph.active_relations.tolist():
            relation_rows = encoded[graph.relation == relation]
            if self.usage_aggregation == "max":
                value = relation_rows.amax(dim=0)
            else:
                rel = r[relation].expand(len(relation_rows), -1)
                logits = self.usage_attention(torch.cat([relation_rows, rel], -1)).flatten()
                value = (torch.softmax(logits, dim=0)[:, None] * relation_rows).sum(0)
            usage[relation] = value
        return usage

    def _transport(self, x, slot, active, coefficients, operators):
        if coefficients is None:
            return x
        if operators is not None:
            return F.linear(x, operators[slot])
        relation = int(active[slot])
        # (n, d) @ (d, p) -> (n, p), then back to (n, d).
        hidden = (x @ self.right[relation]) * coefficients[slot]
        return hidden @ self.left[relation].transpose(0, 1)

    def forward(self, time, state, graph):
        self.nfe += 1
        h, r = state
        if self.variant in {"no_relation_evolution", "joint_ablation"}:
            dr = torch.zeros_like(r)
        else:
            usage = self._usage_summary(h, r, graph)
            code = self.time_code(time).expand(len(r), -1)
            dr = self.relation_flow(torch.cat([r, usage, code], dim=-1))
        dh = torch.zeros_like(h)
        if self.variant == "no_graph_flow" or not graph.num_edges:
            return dh, dr
        active = graph.active_relations
        coefficients = operators = None
        if self.variant not in {"identity_transport", "joint_ablation"}:
            if self.variant == "static_transport_coefficients":
                coefficients = torch.tanh(self.static_coefficients[active])
            else:
                coefficients = self.transport_coefficients(r[active])
            if self.transport_implementation == "dense":
                operators = torch.bmm(
                    self.left[active] * coefficients[:, None, :],
                    self.right[active].transpose(1, 2),
                )
        messages = torch.zeros_like(h)
        for slot, group in enumerate(graph.groups):
            source, destination = h[group.source], h[group.destination]
            if self.transport_formula == "paper":
                # Eq. (15): T_r(t) h_j(t) - h_i(t)
                transported = self._transport(
                    destination, slot, active, coefficients, operators
                ) - source
            else:
                # Historical implementation, retained solely for old checkpoints:
                # T_r(t) (h_j(t) - h_i(t)).
                transported = self._transport(
                    destination - source, slot, active, coefficients, operators
                )
            compatibility = (self.query(source) * self.key(destination)).sum(
                -1
            ) / math.sqrt(self.query.out_features)
            if self.variant == "no_scalar_compatibility":
                compatibility = torch.ones_like(compatibility)
            if self.variant == "no_vector_gate":
                gate = torch.ones_like(source)
            else:
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
