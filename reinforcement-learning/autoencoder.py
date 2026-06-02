"""
Autoencoder for compressing the NVDA feature set into a low-dimensional
latent state for the RL agent.

The model is trained only on the training split. The fitted encoder then
transforms both train and test data, so the test period never influences
the learned representation.
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
from torch import nn

torch.set_num_threads(1)


class Autoencoder(nn.Module):
    def __init__(self, n_in, n_latent):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_in, 16),
            nn.Tanh(),
            nn.Linear(16, 8),
            nn.Tanh(),
            nn.Linear(8, n_latent),
        )
        self.decoder = nn.Sequential(
            nn.Linear(n_latent, 8),
            nn.Tanh(),
            nn.Linear(8, 16),
            nn.Tanh(),
            nn.Linear(16, n_in),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


def train_autoencoder(X_train, n_latent, n_epochs=300, batch_size=256, lr=1e-3, seed=0):
    """Train an autoencoder on X_train and return (model, per-epoch losses)."""
    torch.manual_seed(seed)

    X_train = torch.as_tensor(np.asarray(X_train), dtype=torch.float32)
    n, n_features = X_train.shape

    model = Autoencoder(n_features, n_latent)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    losses = []
    for _ in range(n_epochs):
        perm = torch.randperm(n)
        epoch_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            xb = X_train[idx]
            x_hat, _ = model(xb)
            loss = loss_fn(x_hat, xb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * xb.size(0)
        losses.append(epoch_loss / n)
    return model, losses


def encode(model, X):
    """Return the latent codes for X as a numpy array."""
    model.eval()
    X = torch.as_tensor(np.asarray(X), dtype=torch.float32)
    with torch.no_grad():
        _, z = model(X)
    return z.numpy()


def reconstruction_mse(model, X):
    """Return the mean squared reconstruction error of the model on X."""
    model.eval()
    X = torch.as_tensor(np.asarray(X), dtype=torch.float32)
    with torch.no_grad():
        x_hat, _ = model(X)
    return ((x_hat - X) ** 2).mean().item()
