#!/usr/bin/env python3
import os
import h5py
import pygame
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import Dataset

from utils import *
from config import *


class CommaDataset(Dataset):
  def __init__(self, base_dir):
    super(CommaDataset, self).__init__()
    self.base_dir = base_dir
    self.cam_path = os.path.join(self.base_dir, "camera")
    self.log_path = os.path.join(self.base_dir, "log")
    assert len(os.listdir(self.cam_path)) == len(os.listdir(self.log_path))
    
    self.dataset_length = 0
    self.init_datasets()

  def init_datasets(self):
    print("[*] Initializing dataset")
    self.indices = []
    self.datasets = sorted(os.listdir(self.cam_path))
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
    print(self.indices)
    print(self.dataset_length)
    print("[+] Dataset initialized")

  def __len__(self):
    return self.dataset_length

  def __getitem__(self, index):
    print(index)
    assert 0 <= index < self.dataset_length
    dataset_idx = 0
    data_idx = index
    for length in self.indices:
      if index <= length:
        break
      dataset_idx += 1
      data_idx = index - length - 1
    print(f"{dataset_idx} - {data_idx}")

    log = self.logs[dataset_idx]
    cam = self.cams[dataset_idx]
    angle_steers = log['steering_angle'][data_idx]
    speed_ms = log['speed'][data_idx]
    image = cam['X'][data_idx]
    return {"angle_steers": angle_steers, "speed_ms": speed_ms, "image": image}

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
  data = dataset[111713 + 2]
  print(data)
  data = dataset[len(dataset) - 1]
