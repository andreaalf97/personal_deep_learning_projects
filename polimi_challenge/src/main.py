import torch

from polimi_challenge.src.dataset.debug import show_sample, calc_range, get_mean_std, get_avg_dimensions, \
    batch_statistics, visualize_kernels
from polimi_challenge.src.dataset.mask_data import MaskDataset, get_train_loader
from polimi_challenge.src.params import DATASET_DIR, DEVICE
from polimi_challenge.src.dataset.transforms import COMPOSED_TRANSFORM
from polimi_challenge.src.model.net import Resnet
from polimi_challenge.src.training.train import overfit_batch

if __name__ == '__main__':

    torch.manual_seed(0)

    # dataset = MaskDataset(
    #         DATASET_DIR,
    #         transform=COMPOSED_TRANSFORM,
    #         train=True
    #     )
    #
    # train_loader = get_train_loader(dataset)

    # mean, std = get_mean_std(train_loader)
    # print("DATA MEAN: {}\nDATA STD: {}".format(mean, std))

    # show_sample(dataset, denorm=True, mean=0.5, std=0.3)

    # h, v = get_avg_dimensions(dataset)
    # print("IMAGES AVERAGE DIMENSIONS")
    # print("Horizontal {}\nVertical {}".format(h, v))

    net = Resnet()
    net.load_state_dict(torch.load("overfitted_net.pt"))
    net.to(DEVICE)

    sample = torch.load("default_test_sample.pt")
    images = sample["image"].to(DEVICE)
    labels = sample["label"].to(DEVICE)

    out = net(images)

    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(out, labels)
    print("LOSS:", loss)

    # with torch.no_grad():
    #     # batch_statistics("INPUT BATCH", images)
    #     visualize_kernels(net)
    #     for index, label in enumerate(labels):
    #         _, predicted = torch.max(out, 1)
    #         print("Predicion with label {}: {}".format(label.item(), predicted[index]))
    #
    # # batch_statistics("INPUT BATCH", images)
    # #
    # overfit_batch(net, sample, iterations=1500)
    #
    # torch.save(net.state_dict(), "overfitted_net.pt")
