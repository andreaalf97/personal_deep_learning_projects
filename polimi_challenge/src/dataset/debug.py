from math import sqrt
from os import path, listdir
from PIL import Image
from numpy import array
import torch
import random
from torch.utils.data import DataLoader, Dataset
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


def show_sample(dataset: Dataset, denorm=True, mean=0.0, std=0.0) -> None:
    rand_int = int(random.random()*(len(dataset)))
    image = dataset[rand_int]["image"]  # Returns the first image of a random batch
    shape = list(image.shape)
    if denorm:
        image = image * std + mean  # De-normalize the images (approximately)
    plt.imshow(image.permute(1, 2, 0))
    plt.title("Class {} | Image shape {}".format(str(dataset[rand_int]["label"].item()), shape))
    plt.show()
    return


def get_mean_std(loader: DataLoader) -> (tuple, tuple):

    print("Returning pre-compiled results, comment this out for updating")
    return (0.5695, 0.5194, 0.4885), (0.3046, 0.2972, 0.3005)

    channels_sum = 0.0
    channels_sq_sum = 0.0
    num_batches = 0

    for i, batch in enumerate(loader):
        image = batch["image"]
        channels_sum += torch.mean(image, dim=[0, 2, 3])
        channels_sq_sum += torch.mean(image**2, dim=[0, 2, 3])
        num_batches += 1

        if i % int(len(loader) / 10) == 0:
            print("ON BATCH {} of {}".format(i, len(loader)))

    mean = channels_sum/num_batches
    std = (channels_sq_sum/num_batches - mean**2)**0.5

    return mean, std


def get_avg_dimensions(dataset: Dataset):

    print("Returning pre-calculated dimensions, uncomment to run again")
    return [411.0659, 612.0], [612.0, 441.4939]

    print("CALCULATING AVERAGE DATASET DIMENSIONS")

    # [tot_height, tot_width]
    horizontal = [0.0, 0.0]
    num_h = 0
    vertical = [0.0, 0.0]
    num_v = 0

    for i, sample in enumerate(dataset):

        if i % (int(len(dataset) / 10)) == 0:
            print("On image {} over {}".format(i, len(dataset)))

        image = sample["image"]
        height = list(image.shape)[1]
        width = list(image.shape)[2]
        if width > height: # Horizontal
            horizontal[0] += height
            horizontal[1] += width
            num_h += 1
        else:
            vertical[0] += height
            vertical[1] += width
            num_v += 1

    horizontal = [i/num_h for i in horizontal]
    vertical = [i/num_v for i in vertical]

    return horizontal, vertical


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


def batch_statistics(batch_name: str, batch: torch.Tensor) -> None:
    with torch.no_grad():
        print("[BATCH STATISTICS]", batch_name)
        print("Batch shape --> {}".format(batch.shape))
        print("Batch mean: {:.4f}\tBatch std: {:.4f}".format(torch.mean(batch), torch.std(batch)))
        print("Batch range: ({:.4f}, {:.4f})".format(batch.min(), batch.max()))
        show_batch(batch)


def show_batch(batch: torch.Tensor) -> None:

    if len(batch[0]) == 3 and len(batch) == 4:
        fig = plt.figure()
        fig.suptitle("Batch samples ({})".format(list(batch[0].shape)))
        denormed_batch = batch*0.3 + 0.5
        for i, image in enumerate(denormed_batch):
            fig.add_subplot(2, 2, i+1)
            plt.imshow(image.permute(1, 2, 0).cpu())
        plt.show()
        return

    sample = batch[0]  # The first dimension is just the batch size and we only want one sample
    if len(sample) < 9:
        print("Cannot show batches with less than 9 channels that are not RGB")

    channels = [i for i in range(len(sample))]
    rand_channels = random.sample(channels, 9)

    fig = plt.figure()
    fig.suptitle("Batch sample ({})".format(list(sample.shape)))
    for i, picked_channel in enumerate(rand_channels):
        img = sample[picked_channel]
        fig.add_subplot(3, 3, i+1)
        plt.title("Channel " + str(picked_channel))
        plt.imshow(img.cpu())

    plt.show()


def visualize_kernels(net: torch.nn.Module):
    print("[VISUALIZING KERNELS]")
    with torch.no_grad():
        print("MODULES:")
        print("Kernels sizes: [out_channels, in_channels, height, width]")
        for i, module in enumerate(net.modules()):
            print(i, "-", type(module))
            if isinstance(module, torch.nn.Conv2d):
                weights = module.weight.data.cpu()
                print("Conv2d {} has weight shape {}".format(i, weights.shape))

                all_kernel_indices = [i for i in range(len(weights))]
                sampled_kernel_indices = random.sample(all_kernel_indices, 9)

                fig = plt.figure()
                fig.suptitle("Net module #{} (Conv2d) Kernels size ({}, {})".format(i, len(weights[0][0]), len(weights[0][0][0])))
                for i, index in enumerate(sampled_kernel_indices):
                    sampled_kernel = weights[index]
                    if len(sampled_kernel) == 3:  # If kernel is RGB
                        min_value = sampled_kernel.min()
                        norm_sampled_kernel = (sampled_kernel - min_value) / (sampled_kernel.max() - min_value)
                        fig.add_subplot(3, 3, i+1)
                        plt.imshow(norm_sampled_kernel.permute(1, 2, 0))
                        plt.title("Kernel {} of {}".format(index, len(weights)))
                    else:
                        rand_int = int(random.random()*len(sampled_kernel))
                        rand_kernel = sampled_kernel[rand_int]  # This should have 2 dimensions which are equal to each other
                        min_value = rand_kernel.min()
                        norm_rand_kernel = (rand_kernel - min_value) / (rand_kernel.max() - min_value)
                        fig.add_subplot(3, 3, i+1)
                        plt.imshow(norm_rand_kernel)
                        plt.title("Kernel {}.{} of {}".format(index, rand_int, len(weights)))

                plt.show()
