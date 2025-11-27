#!/usr/bin/env python3
import os
import psutil
import torch
from torch.utils.data import DataLoader

from config import *
from dataset import CommaDataset
from vqvae import VQVAE
from trainer import Trainer

# EXAMPLE USAGE: MODEL_PATH=checkpoints/vqvae.pt CHECKPOINT=checkpoints/vqvae_best.py ./train.py

MODEL_PATH = os.getenv("MODEL_PATH", "checkpoints/vqvae.pt")
CHECKPOINT = os.getenv("CHECKPOINT", None)
WRITER_PATH = os.getenv("WRITER_PATH", None)
CACHE = os.getenv("CACHE", False)
READ_FROM_CACHE = os.getenv("READ_FROM_CACHE", True)

N_WORKERS = psutil.cpu_count(logical=False)
PREFETCH_FACTOR = psutil.cpu_count(logical=False) // 2
PIN_MEMORY = not EMA

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
torch.set_warn_always(False)


if __name__ == "__main__":
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print("[+] Using device:", device)

  os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
  print("\n[*] Configuration:")
  print(f"Model path: {MODEL_PATH}")
  print(f"Checkpoint path: {CHECKPOINT}")
  print(f"Epochs: {EPOCHS} - Batch size: {BATCH_SIZE} - Learning rate: {LR} - Weight decay: {WEIGHT_DECAY}")
  print(f"Number of workers: {N_WORKERS} - Prefetch factor: {PREFETCH_FACTOR}")
  print(f"EMA: {EMA} - Pin memory: {PIN_MEMORY}")
  print(f"Cache dataset: {CACHE} - Read from cache: {READ_FROM_CACHE}")
  # print(f"NORMALIZE_STATES: {NORMALIZE_STATES}")
  print()

  dataset = CommaDataset(BASE_DIR, cache=CACHE, read_from_cache=READ_FROM_CACHE)
  train_split = int(0.8 * len(dataset))
  val_split = int(len(dataset) - train_split)
  train_set, val_set = torch.utils.data.random_split(dataset, [train_split, val_split])

  vqvae = VQVAE(in_size=3, out_size=3).to(device)

  train_loader =  DataLoader(
    train_set, batch_size=BATCH_SIZE, shuffle=True,
    prefetch_factor=PREFETCH_FACTOR, num_workers=N_WORKERS, pin_memory=PIN_MEMORY
  )
  val_loader = DataLoader(
    val_set, batch_size=BATCH_SIZE, shuffle=False,
    prefetch_factor=PREFETCH_FACTOR, num_workers=N_WORKERS, pin_memory=PIN_MEMORY
  )

  trainer = Trainer(
    device, vqvae, MODEL_PATH, train_loader, val_loader,
    checkpoint_path=CHECKPOINT, writer_path=WRITER_PATH, eval_epoch=True,
    save_checkpoints=True, early_stopping=True
  )
  trainer.train()

