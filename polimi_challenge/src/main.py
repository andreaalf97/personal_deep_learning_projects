import torch

from polimi_challenge.src.dataset.debug import show_sample, calc_range, get_mean_std, get_avg_dimensions, \
    batch_statistics, visualize_kernels
from polimi_challenge.src.dataset.mask_data import MaskDataset, get_train_loader
from polimi_challenge.src.params import DATASET_DIR, DEVICE
from polimi_challenge.src.dataset.transforms import COMPOSED_TRANSFORM
from polimi_challenge.src.model.net import Resnet


if __name__ == '__main__':

    dataset = MaskDataset(
            DATASET_DIR,
            transform=COMPOSED_TRANSFORM,
            train=True
        )

    train_loader = get_train_loader(dataset)

    # mean, std = get_mean_std(train_loader)
    # print("DATA MEAN: {}\nDATA STD: {}".format(mean, std))

    # show_sample(dataset, denorm=True, mean=0.5, std=0.3)

    # h, v = get_avg_dimensions(dataset)
    # print("IMAGES AVERAGE DIMENSIONS")
    # print("Horizontal {}\nVertical {}".format(h, v))

    net = Resnet()
    net.to(DEVICE)

    sample = torch.load("default_test_sample.pt")

    images = sample["image"].to(DEVICE)
    labels = sample["label"].to(DEVICE)
    print("INPUT SHAPE: {}".format(images.shape))

    out = net(images)

    with torch.no_grad():
        batch_statistics("INPUT BATCH", images)
        batch_statistics("OUTPUT BATCH", out)
        # visualize_kernels(net)
