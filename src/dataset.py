#!/usr/bin/env python3
import os
import h5py
import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
import torchvision.transforms as T
from torch.utils.data import Dataset

from config import *
from image_tokenizer.constants import *



class CommaDataset(Dataset):
  def __init__(
    self,
    base_dir,
    seq_len,
    normalize_logs=True,
    cache=True,
    read_from_cache=True,
    n_datasets=list(range(N_DATASETS)),
    mode=DataMode.VAL,
    single_frame=False,
    tokens_only=False
  ):
    super(CommaDataset, self).__init__()
    self.base_dir = base_dir
    self.seq_len = seq_len
    self.normalize_logs = normalize_logs
    self.cache = cache
    self.read_from_cache = read_from_cache
    self.n_datasets = n_datasets
    self.mode = mode
    self.single_frame = single_frame
    self.tokens_only = tokens_only

    self.token_dir = os.path.join(self.base_dir, "tokens", self.mode)  # TODO: load tokens
    self.cam_path = os.path.join(self.base_dir, "camera")
    self.log_path = os.path.join(self.base_dir, "log")
    assert len(os.listdir(self.cam_path)) == len(os.listdir(self.log_path))

    self.transform = T.Compose([
      T.Resize((H, W)),
      T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    
    self.dataset_length = 0
    self.init_datasets()
    if os.path.exists(self.token_dir) and self.tokens_only: self.load_tokens()
    self.init_sequences()
    if self.normalize_logs: self.init_logs()

    print("[+] Dataset initialized")

  def load_tokens(self):
    self.all_tokens = []
    self.token_indices = []
    token_batches = [os.path.join(self.token_dir, dirr) for dirr in sorted(os.listdir(self.token_dir)) if dirr.endswith(".npz")]

    for batch in tqdm(token_batches, desc="Loading tokens"):
      data = np.load(batch, allow_pickle=True)
      tokens = data["tokens"]          # shape: (B, Ht, Wt), dtype: uint16
      dataset_idx = data["dataset_idx"]
      inner_idx = data["inner_idx"]
      self.Ht, self.Wt = data["token_hw"]      # [Ht, Wt]
      self.all_tokens.append(tokens)
      self.token_indices.append(np.stack([dataset_idx, inner_idx], axis=1))   # shape: (B, 2)

    self.all_tokens = np.concatenate(self.all_tokens, axis=0)   # shape: (N, Ht, Wt)
    self.token_indices = np.concatenate(self.token_indices, axis=0)   # shape: (N, 2)

  def load_cache(self, base_cache_path):
    self.indices = list(np.load(f"{base_cache_path}/indices.npy"))
    self.datasets = list(np.load(f"{base_cache_path}/datasets.npy"))
    self.cams = list(np.load(f"{base_cache_path}/cams.npy", allow_pickle=True))

    self.steers = np.load(f"{base_cache_path}/steers.npy")
    self.speeds = np.load(f"{base_cache_path}/speeds.npy")

    self.dataset_length = int(np.load(f"{base_cache_path}/dataset_length.npy"))
    self.num_datasets = int(np.load(f"{base_cache_path}/num_datasets.npy"))

    print("[+] Loaded values from cache")
    print("[*] Dataset indices:", self.indices)
    print("[*] Dataset length:", self.dataset_length)
    print(f"[*] Steering angles {self.steers.shape} - Speeds (ms) {self.speeds.shape}")

  def save_cache(self, base_cache_path):
    np.save(f"{base_cache_path}/indices.npy", np.array(self.indices))
    np.save(f"{base_cache_path}/datasets.npy", np.array(self.datasets))
    np.save(f"{base_cache_path}/cams.npy", np.array(self.cams))
    np.save(f"{base_cache_path}/steers.npy", np.array(self.steers))
    np.save(f"{base_cache_path}/speeds.npy", np.array(self.speeds))
    np.save(f"{base_cache_path}/dataset_length.npy", np.array(self.dataset_length))
    np.save(f"{base_cache_path}/num_datasets.npy", np.array(len(self.datasets)))
    print(f"[+] Caches saved in {base_cache_path}/ directory")

  def init_datasets(self):
    print(f"[~] Initializing {self.mode} dataset")
    base_cache_path = f"cache/{self.mode}"
    if self.cache: os.makedirs(base_cache_path, exist_ok=True)

    self.steers, self.speeds = [], []
    if self.read_from_cache and (
      os.path.exists(f"{base_cache_path}/indices.npy") and
      os.path.exists(f"{base_cache_path}/datasets.npy") and
      os.path.exists(f"{base_cache_path}/cams.npy") and
      os.path.exists(f"{base_cache_path}/steers.npy") and
      os.path.exists(f"{base_cache_path}/speeds.npy") and
      os.path.exists(f"{base_cache_path}/dataset_length.npy") and
      os.path.exists(f"{base_cache_path}/num_datasets.npy")
    ):
      self.load_cache(base_cache_path)
      return

    dataset_files = sorted(os.listdir(self.cam_path))
    self.datasets = [dataset_files[i] for i in self.n_datasets]

    self.indices = [] # indices of last element of each dataset
    self.cams = []
    steers_all, speeds_all = [], []
    for i, dataset in enumerate((t := tqdm(self.datasets))):
      # handle cams
      cam_path = os.path.join(self.cam_path, dataset)
      with h5py.File(cam_path, "r") as cam:
        data_len = cam["X"].shape[0]
      self.cams.append(cam_path)
      t.set_description(f"{i+1}/{len(self.datasets)} - {data_len}")

      # dataset lengths and indices
      self.dataset_length += data_len
      if i == 0:
        self.indices.append(data_len - 1)
      else:
        self.indices.append(self.indices[i-1] + data_len - 1)

      # handle logs
      log_file = h5py.File(os.path.join(self.log_path, dataset), "r")

      # CAMS: 20Hz, LOGS: 100Hz
      ratio = log_file['steering_angle'].shape[0] // data_len
      assert ratio > 0, "Invalid log/cam ratio"
      usable_log_len = data_len * ratio  # trim logs to be divisible

      steers = log_file["steering_angle"][:usable_log_len]
      speeds = log_file["speed"][:usable_log_len]
      steers = steers.reshape(data_len, ratio).mean(axis=1)
      speeds = speeds.reshape(data_len, ratio).mean(axis=1)
      steers_all.append(steers)
      speeds_all.append(speeds)
    self.steers = np.concatenate(steers_all)
    self.speeds = np.concatenate(speeds_all)

    if self.cache: self.save_cache(base_cache_path)
    print("[*] Dataset indices:", self.indices)
    print("[*] Dataset length:", self.dataset_length)
    print(f"[*] Steering angles {self.steers.shape} - Speeds (ms) {self.speeds.shape}")

  def init_sequences(self):
    """
    Prepare autoregressive sequences.
    Each sequence is (dataset_idx, start_frame_idx)
    such that:
      context: [start : start + N_FRAMES]
      target : start + N_FRAMES
    """
    print("[~] Initializing sequences")
    self.sequences = []
    prev_dataset_end = -1
    for dataset_idx, dataset_end in enumerate(self.indices):
      dataset_start = prev_dataset_end + 1
      dataset_len = dataset_end - dataset_start + 1

      # number of valid autoregressive sequences in dataset
      max_start = dataset_len - (self.seq_len + 1)
      if max_start < 0:
        prev_dataset_end = dataset_end
        continue

      for start in range(max_start + 1):
        self.sequences.append((dataset_idx, start))

      prev_dataset_end = dataset_end

    print(f"[+] Built {len(self.sequences)} sequences")

  def init_logs(self, zero_to_one=True):
    print("[~] Initializing state values")

    steers_clipped = np.clip(self.steers, -STEER_CLIP, STEER_CLIP)
    speeds_clipped = np.clip(self.speeds, SPEED_MIN, SPEED_MAX)

    # Min-max normalization with safety
    steers_range = steers_clipped.max() - steers_clipped.min()
    speeds_range = speeds_clipped.max() - speeds_clipped.min()
    self.steers = (steers_clipped - steers_clipped.min()) / (steers_range if steers_range != 0 else 1)
    self.speeds = (speeds_clipped - speeds_clipped.min()) / (speeds_range if speeds_range != 0 else 1)

    # [-1, 1]
    if not zero_to_one:
      self.steers = 2 * self.steers - 1
      self.speeds = 2 * self.speeds - 1

    print(f"[*] Steers normalized: min={self.steers.min()}, max={self.steers.max()}")
    print(f"[*] Speeds normalized: min={self.speeds.min()}, max={self.speeds.max()}")

  def __len__(self):
    return len(self.sequences)

  def __del__(self):
    if hasattr(self, "_cam_files"):
      for f in self._cam_files.values():
        f.close()

  def singleframe_getitem(self, index):
    """Fetches a single frame from the dataset. Used for pretraining the image tokenizer"""

    dataset_idx, inner_idx = self.sequences[index]
    cam = self._get_cam(dataset_idx)
    image = self._apply_transform(cam["X"][inner_idx])
    return {
      "index": index,
      "dataset_idx": dataset_idx,
      "inner_idx": inner_idx,
      "image": image,
      # "disp_image": disp_image  # TODO: return displayable image as well
    }

  def multiframe_getitem(self, index):
    """Fetches a sequence of frames and their corresponding actions (steer, speed)"""

    dataset_idx, start_idx = self.sequences[index]

    cam = self._get_cam(dataset_idx)
    images = cam["X"][start_idx : start_idx + self.seq_len + 1]
    seq_frames = self._apply_transform(images[:self.seq_len])
    next_frame = self._apply_transform(images[self.seq_len:self.seq_len + 1])

    gidx = self._global_idx(dataset_idx, start_idx + self.seq_len)
    angle_steers = torch.tensor(self.steers[gidx:gidx+self.seq_len], dtype=torch.float32)
    speed_ms = torch.tensor(self.speeds[gidx:gidx+self.seq_len], dtype=torch.float32)
    actions = torch.stack((angle_steers, speed_ms), dim=1)

    return {
      "seq_frames": seq_frames,
      "next_frame": next_frame,
      "actions": actions  # (steer, speed)
    }

  def token_getitem(self, index):
    """ Fetches a sequence of tokens and their corresponding actions (steer, speed) """
    dataset_idx, start_idx = self.sequences[index]
    seq_tokens = self.all_tokens[start_idx : start_idx + self.seq_len]  # (T, Ht, Wt)
    next_token = self.all_tokens[self.seq_len: self.seq_len + 1]  # (1, Ht, Wt)

    gidx = self._global_idx(dataset_idx, start_idx + self.seq_len)
    angle_steers = torch.tensor(self.steers[gidx:gidx+self.seq_len], dtype=torch.float32)
    speed_ms = torch.tensor(self.speeds[gidx:gidx+self.seq_len], dtype=torch.float32)
    actions = torch.stack((angle_steers, speed_ms), dim=1)

    return {
      "seq_tokens": seq_tokens,
      "next_token": next_token,
      "actions": actions  # (steer, speed)
    }

  def __getitem__(self, index):
    if self.single_frame:
      return self.singleframe_getitem(index)
      
    if self.tokens_only:
      return self.token_getitem(index)

    return self.multiframe_getitem(index)

  def _global_idx(self, dataset_idx, local_idx):
    if dataset_idx == 0:
        return local_idx
    return self.indices[dataset_idx - 1] + 1 + local_idx

  def _apply_transform(self, frames):
    return self.transform(torch.from_numpy(frames).float() / 255.0) # (T, C, H, W)

  def _get_cam(self, dataset_idx):
    if not hasattr(self, "_cam_files"):
      self._cam_files = {}

    if dataset_idx not in self._cam_files:
      self._cam_files[dataset_idx] = h5py.File(self.cams[dataset_idx], "r", swmr=True, libver="latest")

    return self._cam_files[dataset_idx]

  @staticmethod
  def _denormalize_img(image):
    """
    image: (C,H,W)
    returns: HWC uint8
    """
    mean = np.array(IMAGENET_MEAN).reshape(3,1,1)
    std = np.array(IMAGENET_STD).reshape(3,1,1)

    img = image.numpy() * std + mean
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)
    return np.transpose(img, (1,2,0))  # (H, W, C)

  # TODO: write this
  @staticmethod
  def _denormalize_controls(steer, speed):
    pass


if __name__ == "__main__":
  dataset = CommaDataset(
    BASE_DIR,
    N_FRAMES,
    n_datasets=TRAIN_DATASETS,
    cache=True,
    read_from_cache=False,
    tokens_only=True,
    mode=DataMode.TRAIN
  )
  print(len(dataset))
  data = dataset[1000]
  # print(data["seq_frames"].shape)
  # print(data["next_frame"].shape)
  print(data["seq_tokens"].shape)
  print(data["next_token"].shape)
  print(data["actions"])
