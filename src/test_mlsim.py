#!/usr/bin/env python3
import torch

from config import *
from models.mlsim import MLSim, MLSimConfig

def test_model():
  tokens_only = True

  cfg = MLSimConfig()
  model = MLSim(cfg, tokens_only=tokens_only, device=device).to(device)

  if tokens_only:
    x = torch.randint(
      low=0,
      high=model.vqvae_cfg.n_embeddings,
      size=(1, N_FRAMES, 40, 80),
      device=device,
      dtype=torch.long
    )
  else:
    x = torch.randn(1, N_FRAMES, 3, H, W, dtype=torch.float, device=device)
  actions = torch.rand((1, N_FRAMES, 2), device=device)

  out = model(x, actions)
  print(out.shape)


if __name__ == "__main__":
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  test_model()
  print("[+] MLSim model OK")
