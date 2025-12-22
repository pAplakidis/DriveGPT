#!/usr/bin/env python3
import os
import psutil
import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import CommaDataset
from image_tokenizer.config import *
from image_tokenizer.constants import *
from image_tokenizer.vqvae.models.vqvae import VQVAE


MODEL_PATH = "image_tokenizer/checkpoints/compression_1280_bits/vqvae.pt"
BASE_DIR = "../data/comma.ai/comma_research_dataset/comma-dataset/comma-dataset"
BATCH_SIZE = 512

CACHE = os.getenv("CACHE", False)
READ_FROM_CACHE = os.getenv("READ_FROM_CACHE", False)
N_WORKERS = psutil.cpu_count(logical=False)
PREFETCH_FACTOR = psutil.cpu_count(logical=False) // 2
PIN_MEMORY = True

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
torch.set_warn_always(False)


def reconstruct_tokens(tokens, original_img, i_batch, dataset, model):
  """
  tokens: torch.LongTensor or np.ndarray of shape (B, Ht, Wt)
  """

  model.eval()
  device = next(model.parameters()).device

  if isinstance(tokens, np.ndarray):
    tokens = torch.from_numpy(tokens)
  tokens = tokens.to(device).long()  # (B, Ht, Wt)

  B, Ht, Wt = tokens.shape
  n_embeddings = model.vector_quantization.n_e
  embedding_dim = model.vector_quantization.e_dim

  with torch.no_grad():
    # 1️⃣ tokens → one-hot
    one_hot = torch.nn.functional.one_hot(tokens, num_classes=n_embeddings).float() # (B, Ht, Wt, n_embeddings)

    # 2️⃣ one-hot → embeddings
    z_q = torch.matmul(one_hot, model.vector_quantization.embedding.weight) # (B, Ht, Wt, embedding_dim)

    # 3️⃣ permute for decoder
    z_q = z_q.permute(0, 3, 1, 2).contiguous()  # (B, C, Ht, Wt)

    # 4️⃣ decode
    recon_img = model.decoder(z_q)   # (B, 3, H, W)

  # visualization (first sample)
  recon = dataset._denormalize_img(recon_img[0].cpu())
  orig = dataset._denormalize_img(original_img[0].cpu())

  plt.figure(figsize=(8, 4))

  plt.subplot(1, 2, 1)
  plt.imshow(orig)
  plt.title("Original")
  plt.axis("off")

  plt.subplot(1, 2, 2)
  plt.imshow(recon)
  plt.title("Reconstruction")
  plt.axis("off")

  out_path = f"recon_batch{i_batch}.png"
  plt.tight_layout()
  plt.savefig(out_path, dpi=150)
  plt.close()


def init_tokendir(path, mode):
  os.makedirs(path, exist_ok=True)
  meta = {
    "mode": mode,
    "image_size": [H, W],
    "normalization": "imagenet",
    "vqvae_checkpoint": MODEL_PATH,
  }
  with open(os.path.join(path, "meta.json"), "w") as f:
    json.dump(meta, f, indent=2)

def tokenize_dataset(dataloader, model, mode: DataMode):
  token_dir = os.path.join(BASE_DIR, "tokens", mode)
  init_tokendir(token_dir, mode)

  with torch.no_grad():
    image_tokenizer = nn.Sequential(
      model.encoder,
      model.pre_quantization_conv,
      model.vector_quantization
    )

    for i_batch, sample_batched in enumerate((t := tqdm(dataloader, desc=f"Tokenizing {mode} dataset"))):
      image = sample_batched["image"].to(device)
      _, image_tokens, _, _, _ = image_tokenizer(image)
      loss, z_q, perplexity, min_encodings, min_encoding_indices = image_tokenizer(image)

      # (B, D, Ht, Wt) = (batch, embedding_dim, latent_h, latent_w)
      B, C, Ht, Wt = z_q.shape
      tokens = min_encoding_indices.view(B, Ht, Wt)
      batch_data = {
        "tokens": tokens.cpu().numpy().astype(np.uint16),
        "index": np.array(sample_batched["index"]),
        "dataset_idx": np.array(sample_batched["dataset_idx"]),
        "inner_idx": np.array(sample_batched["inner_idx"]),
        "token_hw": np.array([Ht, Wt]),
      }

      out_path = os.path.join(token_dir, f"batch_{i_batch:06d}.npz")
      np.savez_compressed(out_path, **batch_data)

      # reconstruct_tokens(
      #   tokens[:1], # (1, Ht, Wt)
      #   image[:1],  # original image
      #   i_batch,
      #   train_set,
      #   model
      # )

  print(f"[+] {mode} tokens saved at {token_dir}")


if __name__ == "__main__":
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print("[+] Using device:", device)

  os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
  print("\n[*] Configuration:")
  print(f"Model path: {MODEL_PATH}")
  print(f"Number of workers: {N_WORKERS} - Prefetch factor: {PREFETCH_FACTOR}")
  print(f"Cache dataset: {CACHE} - Read from cache: {READ_FROM_CACHE}")
  # print(f"NORMALIZE_STATES: {NORMALIZE_STATES}")
  print()

  train_set = CommaDataset(
    BASE_DIR,
    N_FRAMES,
    cache=CACHE,
    read_from_cache=READ_FROM_CACHE,
    n_datasets=TRAIN_DATASETS,
    mode=DataMode.TRAIN,
    single_frame=True
  )
  val_set = CommaDataset(
    BASE_DIR,
    N_FRAMES,
    cache=CACHE,
    read_from_cache=READ_FROM_CACHE,
    n_datasets=VAL_DATASETS,
    mode=DataMode.VAL,
    single_frame=True
  )

  train_loader =  DataLoader(
    train_set, batch_size=BATCH_SIZE, shuffle=False,
    prefetch_factor=PREFETCH_FACTOR, num_workers=N_WORKERS, pin_memory=PIN_MEMORY
  )
  val_loader = DataLoader(
    val_set, batch_size=BATCH_SIZE, shuffle=False,
    prefetch_factor=PREFETCH_FACTOR, num_workers=N_WORKERS, pin_memory=PIN_MEMORY
  )

  cfg = VQVAEConfig()
  model = VQVAE(
    cfg.n_hiddens,
    cfg.n_residual_hiddens,
    cfg.n_residual_layers,
    cfg.n_embeddings,
    cfg.embedding_dim,
    cfg.beta
  ).to(device)
  checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
  model.load_state_dict(checkpoint["model"])
  model.eval()

  tokenize_dataset(train_loader, model, DataMode.TRAIN)
  tokenize_dataset(val_loader, model, DataMode.VAL)

