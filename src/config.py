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

# TODO: 1048576 bits latent space => compress
@dataclass
class VQVAEConfig:
  n_hiddens: int = 128
  n_residual_hiddens: int = 32
  n_residual_layers: int = 2
  n_embeddings: int = 512
  embedding_dim: int = 64
  beta: float = 0.25

