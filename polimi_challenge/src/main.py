from polimi_challenge.src.dataset.debug import show_sample, calc_range, get_mean_std
from polimi_challenge.src.dataset.mask_data import MaskDataset, get_train_loader
from polimi_challenge.src.params import COMPOSED_TRANSFORM, DATASET_DIR

if __name__ == '__main__':

    train_loader = get_train_loader(
        MaskDataset(
            DATASET_DIR,
            transform=COMPOSED_TRANSFORM
        )
    )

    # mean, std = get_mean_std(train_loader)
    # print("MEAN: {}\nSTD: {}".format(mean, std))

    show_sample(train_loader)