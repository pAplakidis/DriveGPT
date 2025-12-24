#!/usr/bin/env python3
import torch
import torch.nn as nn

from image_tokenizer.config import *
from image_tokenizer.models.vqvae import VQVAE, Encoder, VQVAEConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_model():
  B = 2
  x = torch.randn(B, 3, H, W, dtype=torch.float, device=device)

  cfg = VQVAEConfig()
  vqvae = VQVAE(cfg).to(device)
  print(vqvae)
  recon_x, embedding_indices = vqvae(x)
  print(recon_x.shape, embedding_indices.shape)

  tokenizer = Encoder(cfg).to(device)
  encoding_indices = tokenizer(x)
  print(encoding_indices.shape)
  return


if __name__ == "__main__":
  with torch.no_grad():
    test_model()
  print("[+] VQVAE model OK")

