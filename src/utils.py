import math
import torch
from tqdm import tqdm

from config import *

@torch.no_grad()
def compute_variance_dataloader(dataloader):
  """
  Compute image variance using streaming statistics (Welford).
  Returns scalar variance across all pixels.
  """
  n = 0
  mean = 0.0
  M2 = 0.0

  for batch in tqdm(dataloader, desc="Computing variance"):
    images = batch["image"]                   # (B, C, H, W), float32 normalized by transform
    images = images.view(images.size(0), -1)  # flatten pixels per batch

    # Batch mean and variance
    batch_mean = images.mean(dim=1) # per-image mean
    batch_var = images.var(dim=1, unbiased=False)

    for m, v, count in zip(batch_mean, batch_var, [images.size(1)] * len(images)):
      count = float(count)

      # Welford's streaming update
      delta = m - mean
      new_n = n + count
      mean += delta * (count / new_n)
      M2 += v * count + delta * delta * n * count / new_n
      n = new_n

  variance = M2 / n
  return float(variance)

def calc_latent_bits(config: VQVAEConfig, H_in: int, W_in: int) -> int:
  """
  Calculate bits used by the latent space for a single image.
  
  Args:
      config: VQVAEConfig object
      H_in: input image height
      W_in: input image width
  
  Returns:
      total bits used by latent tokens
  """
  bits_per_token = math.ceil(math.log2(config.n_embeddings))
  
  # Latent grid size after encoder
  H_latent = H_in // config.stride
  W_latent = W_in // config.stride
  
  n_tokens = H_latent * W_latent
  
  total_bits = n_tokens * bits_per_token
  return total_bits
