import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyIDS(nn.Module):
    """Lightweight IDS model designed for TinyML deployment."""
    def __init__(self, input_dim, n_classes, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(hidden, hidden // 2),
            nn.BatchNorm1d(hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(hidden // 2, hidden // 4),
            nn.BatchNorm1d(hidden // 4),
            nn.ReLU(),

            nn.Linear(hidden // 4, n_classes)
        )

    def forward(self, x):
        return self.net(x)

    def get_embedding(self, x):
        """Returns penultimate layer embedding for analysis."""
        for layer in list(self.net.children())[:-1]:
            x = layer(x)
        return x
