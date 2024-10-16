import torch
import torch.nn as nn
from .bert import SBERT


class SBERTRegression(nn.Module):
    """
    Downstream task: Satellite Time Series Classification
    """

    def __init__(self, sbert: SBERT, num_classes, seq_len):
        super().__init__()
        self.sbert = sbert
        self.max_len = seq_len
        self.pooling = nn.MaxPool1d(self.max_len)
        self.linear = nn.Linear(self.sbert.hidden, num_classes)

    def forward(self, x, doy, mask):
        # dimensions of x: [batch_size, sequence_length, num_classes]
        x = self.sbert(x, doy, mask)
        x = self.pooling(x.permute(0, 2, 1)).squeeze()
        x = self.linear(x)
        return x