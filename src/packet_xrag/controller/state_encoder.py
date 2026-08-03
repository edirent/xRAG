"""Selected-set state summaries for the sequential packet controller."""

import torch
import torch.nn as nn


class SelectedSetStateEncoder(nn.Module):
    def __init__(self, hidden_size=512):
        super().__init__()
        self.hidden_size = hidden_size
        self.empty_mean = nn.Parameter(torch.empty(hidden_size))
        self.empty_max = nn.Parameter(torch.empty(hidden_size))
        nn.init.normal_(self.empty_mean, std=0.02)
        nn.init.normal_(self.empty_max, std=0.02)

    def forward(self, projected_packets, selected_indices):
        if not selected_indices:
            return self.empty_mean, self.empty_max
        selected = projected_packets[selected_indices]
        return selected.mean(dim=0), selected.max(dim=0).values
