import torch
import os
from skimage import io
import json

from polimi_challenge.src.params import BATCH_SIZE, SHUFFLE_DATA


class MaskDataset(torch.utils.data.Dataset):

    def __init__(self, root, transform=None, train=True):
        self.transform = transform

        train_root = os.path.join(root, "training")
        test_root = os.path.join(root, "test")

        json_path = os.path.join(root, "train_gt.json")
        with open(json_path, "r") as file:
            js = json.load(file)

        self.samples = []

        if train:
            class_root = os.path.join(train_root, "0")
            for image_name in os.listdir(class_root):
                self.samples.append(
                    (os.path.join(class_root, image_name), 0)
                )

            class_root = os.path.join(train_root, "1")
            for image_name in os.listdir(class_root):
                self.samples.append(
                    (os.path.join(class_root, image_name), 1)
                )

            class_root = os.path.join(train_root, "2")
            for image_name in os.listdir(class_root):
                self.samples.append(
                    (os.path.join(class_root, image_name), 2)
                )
        else:
            missing_images = 0
            tot = 0
            for image_name in os.listdir(test_root):
                if image_name in js:
                    self.samples.append(
                        (os.path.join(test_root, image_name), js[image_name])
                    )
                else:
                    missing_images += 1
                tot += 1
            print("Missing {} over a total of {}".format(missing_images, tot))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        samples_element = self.samples[index]
        image = io.imread(samples_element[0])

        sample = {
            "image": image,
            "label": samples_element[1]
        }

        if self.transform:
            sample = self.transform(sample)

        return sample


def get_train_loader(dataset: MaskDataset):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=SHUFFLE_DATA
    )
