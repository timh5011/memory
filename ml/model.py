"""Simple MLP for classification with parameter flattening utilities."""

import torch
import torch.nn as nn


class MLP(nn.Module):
    """Multi-layer perceptron with ReLU activations and log-softmax output."""

    def __init__(self, input_dim=2, hidden_dims=None, output_dim=3):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 64]

        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.LogSoftmax(dim=-1))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    def param_count(self):
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters())

    def flat_params(self):
        """Return all parameters as a single 1D tensor."""
        return torch.cat([p.data.reshape(-1) for p in self.parameters()])

    def load_flat_params(self, flat):
        """Load parameters from a flat 1D tensor."""
        offset = 0
        for p in self.parameters():
            n = p.numel()
            p.data.copy_(flat[offset:offset + n].reshape(p.shape))
            offset += n


if __name__ == "__main__":
    model = MLP()
    print(f"Parameter count: {model.param_count()}")

    # Verify flat_params round-trip
    flat = model.flat_params()
    print(f"Flat params shape: {flat.shape}")

    # Perturb and reload
    original = flat.clone()
    model.load_flat_params(flat + 0.01)
    reloaded = model.flat_params()
    assert torch.allclose(reloaded, original + 0.01), "Round-trip failed!"
    print("Flat params round-trip: OK")

    # Test forward pass
    x = torch.randn(10, 2)
    out = model(x)
    print(f"Output shape: {out.shape}")
    print(f"Output sums to ~1 per sample: {torch.exp(out).sum(dim=1)[:3]}")
