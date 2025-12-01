import torch
from tqdm import tqdm

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
