#!/usr/bin/env python3
import torch

from config import *
from models.mlsim import MLSim, MLSimConfig

def test_model():
  images = torch.randn(1, N_FRAMES, 3, H, W, dtype=torch.float, device=device)
  actions = torch.rand((1, N_FRAMES, 2), device=device)
  cfg = MLSimConfig()
  model = MLSim(cfg, device=device).to(device)
  out = model(images, actions)
  print(out.shape)


if __name__ == "__main__":
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  test_model()
  print("[+] MLSim model OK")
