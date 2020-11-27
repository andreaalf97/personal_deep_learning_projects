from os import path, listdir
from PIL import Image
from numpy import array
from torch import tensor
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

from polimi_challenge.src.params import DATASET_DIR

def ds_show_example():

    EXAMPLE_DIR = path.join(DATASET_DIR, "training/0")
    all_image_paths = listdir(EXAMPLE_DIR)[:1]
    all_image_paths = [path.join(EXAMPLE_DIR, i) for i in all_image_paths]

    images = []

    for image_path in all_image_paths:
        im = Image.open(image_path)
        numpy_image = array(im)
        images.append(tensor(numpy_image))

    plt.imshow(make_grid(images))
    plt.title("WITH NO MASK")
    plt.show()


if __name__ == '__main__':
    ds_show_example()