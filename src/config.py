from dataclasses import dataclass

BASE_DIR = "../data/comma.ai/comma_research_dataset/comma-dataset/comma-dataset"

H = 160
W = 320
N_FRAMES = 5

N_DATASETS = 11
TRAIN_DATASETS = list(range(0, 8))
VAL_DATASETS = list(range(8, 10))

N_WORKERS = 8

BATCH_SIZE = 64
EMA = False
LR = 3e-4
WEIGHT_DECAY = 1e-5
EPOCHS = 100
EARLY_STOP_EPOCHS = 15

LR_FACTOR = 0.1
LR_PATIENCE = 10


# Latent Space Size Calculation:
# bits per token=⌈log2​(n_embeddings)⌉ (e.g. n_embeddings = 256 → 8 bits, n_embeddings = 512 → 9 bits)
# number of tokens (latent grid size) = H_latent × W_latent = H_in / s_h × W_in / s_w (e.g. s_h = s_w = 4 → 40 × 80 = 3200, s_h = s_w = 8 → 20 × 40 = 800)
# s_h, s_w: total downsampling factor of encoder in height and width dimensions respectively
# total bits per image = N_tokens × bits per token = (H_in / s_h × W_in / s_w) × ⌈log2​(n_embeddings)⌉

# 256 × 9 = 2304 bits (original config from paper)
# @dataclass
# class VQVAEConfig:
#   n_hiddens: int = 128
#   n_residual_hiddens: int = 32
#   n_residual_layers: int = 2
#   n_embeddings: int = 512
#   embedding_dim: int = 64
#   beta: float = 0.25

# 128 tokens × 10 bits = 1280 bits
@dataclass
class VQVAEConfig:
  n_hiddens: int = 128
  n_residual_hiddens: int = 32
  n_residual_layers: int = 2
  n_embeddings: int = 256
  embedding_dim: int = 32
  beta: float = 0.25
  stride: int = 4 # total downsampling factor of the encoder
