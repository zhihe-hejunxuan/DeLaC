from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class FTCPVAE(nn.Module):
    """Crystal-level convolutional VAE with the original property head."""

    def __init__(self, matrix_shape: tuple[int, int], latent_dim: int = 256,
                 property_dim: int = 0):
        super().__init__()
        self.matrix_shape, self.latent_dim, self.property_dim = matrix_shape, latent_dim, property_dim
        rows, channels = matrix_shape
        self.encoder_conv = nn.Sequential(
            nn.Conv1d(channels, 32, 5, 2, 2), nn.BatchNorm1d(32), nn.LeakyReLU(.2),
            nn.Conv1d(32, 64, 3, 2, 1), nn.BatchNorm1d(64), nn.LeakyReLU(.2),
            nn.Conv1d(64, 128, 3, 1, 1), nn.BatchNorm1d(128), nn.LeakyReLU(.2))
        with torch.no_grad():
            h = self.encoder_conv(torch.zeros(2, channels, rows))
        self.map_len = h.shape[-1]
        self.encoder_dense = nn.Sequential(nn.Flatten(), nn.Linear(128 * self.map_len, 1024), nn.Sigmoid())
        self.mu, self.logvar = nn.Linear(1024, latent_dim), nn.Linear(1024, latent_dim)
        self.decoder_dense = nn.Sequential(nn.Linear(latent_dim, 128 * self.map_len), nn.ReLU())
        length1 = (rows + 1) // 2
        length2 = (length1 + 1) // 2
        output_padding1 = length1 - (2 * length2 - 1)
        output_padding2 = rows - (2 * length1 - 1)
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose1d(128, 64, 3, 1, 1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.ConvTranspose1d(64, 32, 3, 2, 1, output_padding=output_padding1), nn.BatchNorm1d(32), nn.ReLU(),
            nn.ConvTranspose1d(32, channels, 5, 2, 2, output_padding=output_padding2), nn.Sigmoid())
        self.property_head = (nn.Sequential(nn.ReLU(), nn.Linear(latent_dim, 128), nn.ReLU(),
                                             nn.Linear(128, 32), nn.ReLU(),
                                             nn.Linear(32, property_dim), nn.Sigmoid())
                              if property_dim else None)

    def encode(self, x):
        h = self.encoder_dense(self.encoder_conv(x.transpose(1, 2)))
        return self.mu(h), self.logvar(h)

    @staticmethod
    def reparameterize(mu, logvar):
        return mu + torch.exp(.5 * logvar) * torch.randn_like(mu)

    def decode(self, z):
        h = self.decoder_dense(z).view(-1, 128, self.map_len)
        out = self.decoder_conv(h)[..., :self.matrix_shape[0]]
        return out.transpose(1, 2)

    def forward(self, x, sample: bool = True):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar) if sample else mu
        return {"reconstruction": self.decode(z), "mu": mu, "logvar": logvar,
                "property": self.property_head(mu) if self.property_head else None}

    def loss(self, output, target, properties=None, property_mask=None,
             kl_weight=2.0, property_weight=10.0, legacy_keras_scaling=True):
        # Match the original Keras implementation: sum features, average batch.
        recon = (output["reconstruction"] - target).square().flatten(1).sum(1).mean()
        kl = -.5 * (1 + output["logvar"] - output["mu"].square() - output["logvar"].exp()).sum(1).mean()
        prop = target.new_zeros(())
        if self.property_head is not None and properties is not None:
            mask = torch.ones_like(properties, dtype=torch.bool) if property_mask is None else property_mask
            if mask.any():
                prop = (output["property"] - properties).square().masked_select(mask).sum() / target.shape[0]
        # Original Keras summed reconstruction over a batch but averaged KL.
        effective_kl = kl_weight / target.shape[0] if legacy_keras_scaling else kl_weight
        total = recon + effective_kl * kl + property_weight * prop
        return total, {"loss": total.detach(), "recon": recon.detach(), "kl": kl.detach(), "property": prop.detach()}
