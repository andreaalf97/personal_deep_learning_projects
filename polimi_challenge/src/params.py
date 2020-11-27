from torch import device, cuda
from os import path

ROOT_DIR = path.dirname(path.abspath(__file__))
DATASET_DIR = path.join(ROOT_DIR, "..", "Challenge1-dataset")

DEVICE = device("cuda:0" if cuda.is_available() else "cpu")

