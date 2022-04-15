# import the necessary packages
import torch
import os
# base path of the dataset
DATASET_PATH = os.path.join("dataset", "train")
# define the path to the images and masks dataset
IMAGE_DATASET_PATH = os.path.join(DATASET_PATH, "images")
MASK_DATASET_PATH = os.path.join(DATASET_PATH, "masks")

# DATASET_PATH:       dataset/train
# IMAGE_DATASET_PATH: dataset/train/images
# MASK_DATASET_PATH:  dataset/train/masks

# define the test split
TEST_SPLIT_RATIO = 0.15
# determine the device to be used for training and evaluation
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = "cuda:1"
# determine if we will be pinning memory during data loading
PIN_MEMORY = True if DEVICE == "cuda" else False

NUM_CHANNELS = 1
NUM_CLASSES = 1
NUM_LEVELS = 3

INIT_LR = 0.001
NUM_EPOCH = 50
BATCH_SIZE = 64

INPUT_IMAGE_WIDGH = 128
INPUT_IMAGE_HEIGHT = 128

THRESHOLD = 0.5

BASE_OUTPUT = 'output'

MODEL_PATH = os.path.sep.join([BASE_OUTPUT, 'model_names.txt'])
PLOT_PATH = os.path.sep.join([BASE_OUTPUT, 'plot.png'])
TEST_PATHS = os.path.sep.join([BASE_OUTPUT, 'test_paths.txt'])

# MODEL_PATH: output/unet_tgs_salt.pth
# PLOT_PATH:  output/plot.png
# TEST_PATHS: output/test_paths.txt