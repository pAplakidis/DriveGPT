#!/usr/bin/env python3
import torch

from config import *
from models.mlsim import MLSim, MLSimConfig

def test_model():
  tokens_only = True
  B = 2

  cfg = MLSimConfig()
  model = MLSim(cfg, tokens_only=tokens_only, device=device).to(device)

  if tokens_only:
    x = torch.randint(
      low=0,
      high=model.vqvae_cfg.n_embeddings,
      size=(B, N_FRAMES, 40, 80),
      device=device,
      dtype=torch.long
    )
  else:
    x = torch.randn(B, N_FRAMES, 3, H, W, dtype=torch.float, device=device)
  actions = torch.rand((B, N_FRAMES, 2), device=device)
  targets = torch.randint(
    low=0,
    high=model.vqvae_cfg.n_embeddings,
    size=(B, 40, 80),
    device=device,
    dtype=torch.long
  )

  out, loss = model(x, actions, targets=targets)
  print(out.shape)


if __name__ == "__main__":
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  test_model()
  print("[+] MLSim model OK")
