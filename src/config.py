BASE_DIR = "../data/comma.ai/comma_research_dataset/comma-dataset/comma-dataset"

H = 160
W = 320
N_FRAMES = 5

N_DATASETS = 11
TRAIN_DATASETS = list(range(0, 8))
VAL_DATASETS = list(range(8, 10))

BATCH_SIZE = 32
EMA = False
LR = 3e-4
WEIGHT_DECAY = 1e-5
EPOCHS = 100
EARLY_STOP_EPOCHS = 15

LR_FACTOR = 0.1
LR_PATIENCE = 10