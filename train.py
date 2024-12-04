#!/usr/bin/env python3
import os
import torch
from torch.utils.data import DataLoader

from config import *
from dataset import CommaDataset
from model import *

MODEL_PATH = os.getenv("MODEL_PATH")
if MODEL_PATH is None:
  MODEL_PATH = "models/DriveRNN.pt"
print("[+] Model save path:", MODEL_PATH)

CHECKPOINT_PATH = os.getenv("CHECKPOINT_PATH")
FINETUNE = False
if CHECKPOINT_PATH is not None:
  FINETUNE = True
print("[+] Model load path:", CHECKPOINT_PATH)

WRITER_PATH = os.getenv("WRITER_PATH")
if WRITER_PATH is not None:
  print("[+] Tensorboard Writer path:", WRITER_PATH)


if __name__ == "__main__":
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  print("[+] Using device:", device)

  train_set = CommaDataset(BASE_DIR)

  autoencoder = Autoencoder(rows, cols, z_dim).to(device)
  discriminator = Discriminator()
