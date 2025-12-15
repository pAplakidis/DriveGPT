from dataclasses import dataclass

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

STEER_CLIP = 5.0   # radians (TODO: example, choose a realistic vehicle limit)
SPEED_MIN = 0.0
SPEED_MAX = 35.0   # m/s (~126 km/h)

@dataclass
class DataMode:
  TRAIN = "train"
  VAL = "val"
