"""Global full-vocabulary TuckER ranking, independent of graph transport."""

import torch
from torch import nn
from torch.nn import functional as F


def normalize_states(states, norm):
    # BatchNorm needs multiple rows to estimate variance during training.
    if norm.training and len(states) == 1:
        return F.batch_norm(
            states,
            norm.running_mean,
            norm.running_var,
            norm.weight,
            norm.bias,
            training=False,
            eps=norm.eps,
        )
    return norm(states)


class TuckerDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        d, dr = config["entity_dim"], config["relation_dim"]
        self.core = nn.Parameter(torch.empty(dr, d, d))
        nn.init.xavier_uniform_(self.core.view(dr, -1))
        self.entity_norm = nn.BatchNorm1d(d)
        self.relation_norm = nn.BatchNorm1d(dr)
        self.dropout = nn.Dropout(config["dropout"])

    def prepare(self, entities, relations):
        return self.dropout(normalize_states(entities, self.entity_norm)), self.dropout(
            normalize_states(relations, self.relation_norm)
        )

    def forward(self, entities, relations, subjects, relation_ids):
        unique, inverse = torch.unique(relation_ids, sorted=True, return_inverse=True)
        operators = torch.einsum("qr,rij->qij", relations[unique], self.core)
        query = torch.bmm(entities[subjects].unsqueeze(1), operators[inverse]).squeeze(
            1
        )
        return query @ entities.T
