from torch import device, cuda
from os import path
from torchvision.transforms import Compose
from polimi_challenge.src.dataset.transforms import ToTensor, Rescale, RandomCrop

ROOT_DIR = path.dirname(path.abspath(__file__))

DATASET_DIR = path.join(ROOT_DIR, "..", "Challenge1-dataset")
COMPOSED_TRANSFORM = Compose([
        ToTensor(),
        Rescale(400),
        RandomCrop(400)
    ])

BATCH_SIZE = 3
SHUFFLE_DATA = True

DEVICE = device("cuda:0" if cuda.is_available() else "cpu")

