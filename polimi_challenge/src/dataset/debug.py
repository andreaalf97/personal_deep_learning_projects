from math import sqrt
from os import path, listdir
from PIL import Image
from numpy import array
import torch
import random
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

from polimi_challenge.src.params import DATASET_DIR


def ds_show_example():

    example_dir = path.join(DATASET_DIR, "training/0")
    all_image_paths = listdir(example_dir)[:1]
    all_image_paths = [path.join(example_dir, i) for i in all_image_paths]

    images = []

    for image_path in all_image_paths:
        im = Image.open(image_path)
        numpy_image = array(im)
        images.append(torch.tensor(numpy_image))

    plt.imshow(make_grid(images))
    plt.title("WITH NO MASK")
    plt.show()


def data_size(train_loader: DataLoader) -> torch.Size:
    image, label = train_loader[0]["image"], train_loader[0]["label"]
    return image.shape


def show_sample(train_loader: DataLoader) -> None:
    for batch in train_loader:
        image = batch["image"][0]  # Returns the first image of a random batch
        image = image * 0.25 + 0.5  # De-normalize the images (approximately)
        plt.imshow(image.permute(1, 2, 0))
        plt.title("Class " + str(batch["label"][0]))
        plt.show()
        return


def get_mean_std(loader: DataLoader) -> (float, float):
    channels_sum = 0.0
    channels_sq_sum = 0.0
    num_batches = 0

    for i, batch in enumerate(loader):
        image = batch["image"]
        channels_sum += torch.mean(image, dim=[0, 2, 3])
        channels_sq_sum += torch.mean(image**2, dim=[0, 2, 3])
        num_batches += 1

        if i % 100 == 0:
            print("ON BATCH", i)

    mean = channels_sum/num_batches
    std = (channels_sq_sum/num_batches - mean**2)**0.5

    return mean, std


def calc_range(train_loader: DataLoader) -> (float, float):
    glob_min = 1000000000
    glob_max = -1000000000

    i = 0
    for sample in train_loader:
        i += 1
        image, label = sample["image"], sample["label"]

        running_min = torch.min(image)
        running_max = torch.max(image)

        if running_min < glob_min:
            glob_min = running_min

        if running_max > glob_max:
            glob_max = running_max

        if i % 100 == 0:
            print("ON IMAGE", i)

    return float(glob_min.item()), float(glob_max.item())
