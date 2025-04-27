import torch
import torch.nn as nn

class Select(nn.Module):
    """Select a tensor from a list of tensors by index."""
    def __init__(self, idx):
        super().__init__()
        self.idx = idx

    def forward(self, x):
        return x[self.idx]
