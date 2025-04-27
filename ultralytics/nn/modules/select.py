import torch
import torch.nn as nn

class Select(nn.Module):
    """Select a tensor from a list of tensors by index."""
    def __init__(self, idx):
        super().__init__()
        self.idx = idx

    def forward(self, x):
        print(f"Select layer input type: {type(x)}, input length: {len(x) if isinstance(x, list) else 'N/A'}")  # Debug: Print input type and length
        print(f"Select layer selecting index: {self.idx}")  # Debug: Print selected index
        selected = x[self.idx]
        print(f"Select layer output shape: {selected.shape}")  # Debug: Print selected feature map shape
        return selected
