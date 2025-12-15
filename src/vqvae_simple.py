#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.functional as F
from dataclasses import dataclass

from image_tokenizer.config import *


# TODO: use this instead of hardcoding values in model definition
@dataclass
class VQVAEConfig:
  in_size: int = 3
  out_size: int = 3
  hidden_size: int = 16
  n_embeddings: int = 512 # K
  embedding_dim: int = 64 # D
  beta: float = 0.25


class VQVAE(nn.Module):
  def __init__(self, in_size, out_size, hidden_size=4):
    super().__init__()

    self.encoder = Encoder(in_size, hidden_size)
    self.latent_space = VQLatentSpace(hidden_size, hidden_size)
    self.decoder = Decoder(hidden_size, out_size)

  def forward(self, x):
    x = self.encoder(x)
    x, quantize_loss = self.latent_space(x)
    x = self.decoder(x)
    return x, quantize_loss


class Encoder(nn.Module):
  def __init__(self, in_size, out_size, hidden_size=16):
    super().__init__()

    self.encoder = nn.Sequential(
      nn.Conv2d(in_channels=in_size, out_channels=hidden_size, kernel_size=4, stride=2, padding=1),
      nn.BatchNorm2d(hidden_size),
      nn.ReLU(),
      nn.Conv2d(in_channels=hidden_size, out_channels=out_size, kernel_size=4, stride=2, padding=1),
      nn.BatchNorm2d(out_size),
      nn.ReLU(),
    )

  def forward(self, x):
    return self.encoder(x)


class VQLatentSpace(nn.Module):
  def __init__(self, in_size, out_size, hidden_size=2, n_embeddings=3, embedding_dim=2):
    super().__init__()

    self.beta = 0.2

    self.pre_quant_conv = nn.Conv2d(in_channels=in_size, out_channels=hidden_size, kernel_size=1)
    self.embedding = nn.Embedding(num_embeddings=n_embeddings, embedding_dim=embedding_dim)
    self.post_quant_conv = nn.Conv2d(in_channels=hidden_size, out_channels=out_size, kernel_size=1)

  def forward(self, x):
    # Quantization
    quant_input = self.pre_quant_conv(x)
    B, C, H, W = quant_input.shape
    quant_input = quant_input.permute(0, 2, 3, 1)
    quant_input = quant_input.reshape((quant_input.size(0), -1, quant_input.size(-1)))

    # compute pairwise distance argmin of nearest embedding
    dist = torch.cdist(quant_input, self.embedding.weight[None, :].repeat((quant_input.size(0), 1, 1)))
    min_encoding_idxs = torch.argmin(dist, dim=2)

    # select embedding weights
    quant_out = torch.index_select(self.embedding.weight, 0, min_encoding_idxs.view(-1))
    quant_input = quant_input.reshape((-1, quant_input.size(-1)))

    # compute losses
    commitment_loss = torch.mean((quant_out.detach() - quant_input) ** 2)
    codebook_loss = torch.mean((quant_out - quant_input.detach()) ** 2)
    quantize_loss = codebook_loss + self.beta * commitment_loss

    # ensure straight-through gradient and reshape back to original
    quant_out = quant_input + (quant_out - quant_input).detach()
    quant_out = quant_out.reshape((B, H, W, C)).permute(0, 3, 1, 2)
    min_encoding_idxs = min_encoding_idxs.reshape((-1, quant_out.size(-2), quant_out.size(-1)))

    x = self.post_quant_conv(quant_out)

    return x, quantize_loss


class Decoder(nn.Module):
  def __init__(self, in_size, out_size, hidden_size=16):
    super().__init__()

    self.decoder = nn.Sequential(
      nn.ConvTranspose2d(in_channels=in_size, out_channels=hidden_size, kernel_size=4, stride=2, padding=1),
      nn.BatchNorm2d(hidden_size),
      nn.ReLU(),
      nn.ConvTranspose2d(in_channels=hidden_size, out_channels=out_size, kernel_size=4, stride=2, padding=1),
      nn.Tanh()
    )

  def forward(self, x):
    return self.decoder(x)


if __name__ == "__main__":
  model = VQVAE(3, 3)
  t = torch.ones((1, 3, H, W))
  out, loss = model(t)
  print(out.shape)

