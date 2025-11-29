#!/usr/bin/env python3
import os
import h5py
import pygame
import numpy as np
from tqdm import tqdm
from PIL import Image

import torchvision.transforms as transforms
from torch.utils.data import Dataset

from config import *
from constants import *


class CommaDataset(Dataset):
  def __init__(self, base_dir, multiframe=False, cache=True, read_from_cache=True, n_datasets=list(range(N_DATASETS)), mode=DataMode.VAL):
    super(CommaDataset, self).__init__()
    self.base_dir = base_dir
    self.multiframe = multiframe
    self.cache = cache
    self.read_from_cache = read_from_cache
    self.n_datasets = n_datasets
    self.mode = mode

    self.cam_path = os.path.join(self.base_dir, "camera")
    self.log_path = os.path.join(self.base_dir, "log")
    assert len(os.listdir(self.cam_path)) == len(os.listdir(self.log_path))

    transform_list = [transforms.ToTensor(), transforms.Resize((H, W)), transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)]
    self.transform = transforms.Compose(transform_list)
    
    self.dataset_length = 0
    self.init_datasets()

  def init_datasets(self):
    print(f"[*] Initializing {self.mode} dataset")

    if self.cache and (
      os.path.exists(f"cache/{self.mode}/indices.npy") and
      os.path.exists(f"cache/{self.mode}/datasets.npy") and
      os.path.exists(f"cache/{self.mode}/cams.npy") and
      os.path.exists(f"cache/{self.mode}/logs.npy") and
      os.path.exists(f"cache/{self.mode}/dataset_length.npy") and
      os.path.exists(f"cache/{self.mode}/num_datasets.npy")
    ):
      # TODO: train and val folders
      self.indices = list(np.load(f"cache/{self.mode}/indices.npy"))
      self.datasets = list(np.load(f"cache/{self.mode}/datasets.npy"))
      self.cams = list(np.load(f"cache/{self.mode}/cams.npy", allow_pickle=True))
      self.logs = list(np.load(f"cache/{self.mode}/logs.npy", allow_pickle=True))
      self.dataset_length = int(np.load(f"cache/{self.mode}/dataset_length.npy"))
      self.num_datasets = int(np.load(f"cache/{self.mode}/num_datasets.npy"))
      print("[+] Loaded from cache")
    else:
      dataset_files = sorted(os.listdir(self.cam_path))
      self.datasets = [dataset_files[i] for i in self.n_datasets]

      self.indices = []
      self.cams = []
      self.logs = []
      for i, dataset in enumerate((t := tqdm(self.datasets))):
        self.cams.append(cam := h5py.File(os.path.join(self.cam_path, dataset), "r"))
        self.logs.append(h5py.File(os.path.join(self.log_path, dataset), "r"))
        data_len = cam['X'][()].shape[0]
        t.set_description(f"{i+1}/{len(self.datasets)} - {data_len}")

        self.dataset_length += data_len - 1
        if i == 0:
          self.indices.append(data_len - 1)
        else:
          self.indices.append(data_len - 1 + self.indices[i-1])
      self.dataset_length += 1

      if self.cache:
        np.save(f"cache/{self.mode}/indices.npy", np.array(self.indices))
        np.save(f"cache/{self.mode}/datasets.npy", np.array(self.datasets))
        np.save(f"cache/{self.mode}/cams.npy", np.array(self.cams))
        np.save(f"cache/{self.mode}/logs.npy", np.array(self.logs))  # FIXME: ValueError: setting an array element with a sequence. The requested array has an inhomogeneous shape after 1 dimensions. The detected shape was (11,) + inhomogeneous part.
        np.save(f"cache/{self.mode}/dataset_length.npy", np.array(self.dataset_length))
        np.save(f"cache/{self.mode}/num_datasets.npy", np.array(len(self.datasets)))
        print("[+] Caches saved in cache/{self.mode}/ directory")

    print("[*] Dataset indices:", self.indices)
    print("[*] Dataset length:", self.dataset_length)
    print("[+] Dataset initialized")

  def __len__(self):
    return self.dataset_length

  def __getitem__(self, index):
    # assert 0 <= index < (self.dataset_length - 1 - ((N_FRAMES - 1) if self.multiframe else 0))

    dataset_idx = 0
    data_idx = index
    for length in self.indices:
      if index <= length: break
      dataset_idx += 1
      data_idx = index - length - 1
    # print(f"{dataset_idx} - {data_idx}")

    log = self.logs[dataset_idx]
    cam = self.cams[dataset_idx]
    angle_steers = log['steering_angle'][data_idx]
    speed_ms = log['speed'][data_idx]

    # images = cam['X'][data_idx:data_idx+N_FRAMES if self.multiframe else data_idx]
    disp_image = cam['X'][data_idx]
    image = Image.fromarray(np.transpose(disp_image, (1, 2, 0)))
    image = self.transform(image)

    return {"angle_steers": angle_steers, "speed_ms": speed_ms, "image": image, "disp_image": disp_image}

  def init_display(self):
    size = (320*2, 160*2)
    pygame.display.set_caption("comma.ai data viewer")
    self.screen = pygame.display.set_mode(size, pygame.DOUBLEBUF)
    self.camera_surface = pygame.surface.Surface((W,H),0,24).convert()
    print("[+] Pygame display initialized")
  
  def display_img(self, img):
    img = img.swapaxes(0,2).swapaxes(0,1)
    while True:
      pygame.surfarray.blit_array(self.camera_surface, img.swapaxes(0,1))
      camera_surface_2x = pygame.transform.scale2x(self.camera_surface)
      self.screen.blit(camera_surface_2x, (0,0))
      pygame.display.flip()


if __name__ == "__main__":
  dataset = CommaDataset(BASE_DIR)
  print(len(dataset))
  data = dataset[100]
  print(data)
