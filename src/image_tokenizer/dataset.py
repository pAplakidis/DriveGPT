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
    n_datasets=list(range(N_DATASETS)),
    mode=DataMode.VAL,
    single_frame=False,
  ):
    super(CommaDataset, self).__init__()
    self.base_dir = base_dir
    self.n_datasets = n_datasets
    self.mode = mode
    self.single_frame = single_frame

    self.cam_path = os.path.join(self.base_dir, "camera")
    self.transform = T.Compose([
      T.Resize((H, W)),
      T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    
    self.dataset_length = 0
    self.init_datasets()
    print("[+] Dataset initialized")

  def init_datasets(self):
    print(f"[~] Initializing {self.mode} dataset")

    dataset_files = sorted(os.listdir(self.cam_path))
    self.datasets = [dataset_files[i] for i in self.n_datasets]

    self.indices = [] # indices of last element of each dataset
    self.cams = []
    for i, dataset in enumerate((t := tqdm(self.datasets))):
      cam_path = os.path.join(self.cam_path, dataset)
      with h5py.File(cam_path, "r") as cam:
        data_len = cam["X"].shape[0]
      self.cams.append(cam_path)
      t.set_description(f"{i+1}/{len(self.datasets)} - {data_len}")

      self.indices.append(data_len - 1 if i == 0 else self.indices[i - 1] + data_len)
    self.dataset_length = self.indices[-1] + 1

    print("[*] Dataset indices:", self.indices)
    print("[*] Dataset length:", self.dataset_length)

  def __len__(self):
    return self.dataset_length

  def __del__(self):
    if hasattr(self, "_cam_files"):
      for f in self._cam_files.values():
        f.close()

  def __getitem__(self, index):
    # find which dataset this index belongs to and compute index inside that dataset
    dataset_idx = np.searchsorted(self.indices, index, side="left")
    prev_end = self.indices[dataset_idx - 1] if dataset_idx > 0 else -1
    frame_idx = index - prev_end - 1

    cam = self._get_cam(dataset_idx)
    image = self._apply_transform(cam["X"][frame_idx])
    return {
      "index": index,
      "dataset_idx": dataset_idx,
      "inner_idx": frame_idx,
      "image": image,
    }

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
