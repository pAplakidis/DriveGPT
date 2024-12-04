BASE_DIR = "/media/paul/HDD/Datasets/comma.ai/comma_research_dataset/comma-dataset/comma-dataset"

H = 160
W = 320
N_FRAMES = 5

BS = 2
N_WORKERS = 8

image_shape = (3, 80, 160)
ch = image_shape[0]
rows = [image_shape[1]//i for i in [16, 8, 4, 2, 1]]
cols = [image_shape[2]//i for i in [16, 8, 4, 2, 1]]
z_dim = 512 
