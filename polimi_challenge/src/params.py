from torch import device, cuda
from os import path

ROOT_DIR = path.dirname(path.abspath(__file__))

LABEL_MEANING = {
    0: "NO MASK",
    1: "ALL MASKS",
    2: "FEW MASKS"
}

DATASET_DIR = path.join(ROOT_DIR, "..", "Challenge1-dataset")

BATCH_SIZE = 4
DATA_MEAN = (0.5695, 0.5194, 0.4885)
DATA_STD = (0.3046, 0.2972, 0.3005)

RESCALE_TO = (224, 224)

SHUFFLE_DATA = True

DEVICE = device("cuda:0" if cuda.is_available() else "cpu")

