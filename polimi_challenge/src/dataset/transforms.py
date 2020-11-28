import torchvision
import torch
from torchvision.transforms import Compose

from polimi_challenge.src.params import DATA_MEAN, DATA_STD, RESCALE_TO


class Normalize(object):
    """
    This callable class normalizes the images
    """

    def __init__(self, mean, std):
        assert isinstance(mean, tuple)
        assert isinstance(std, tuple)
        assert len(mean) == len(std) and len(mean) == 3
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        image = sample["image"]

        norm = torchvision.transforms.Normalize(self.mean, self.std)
        image = norm(image)

        return {
            "image": image,
            "label": sample["label"]
        }


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image = sample['image']

        # Resizes the smallest edge to output_size
        tr = torchvision.transforms.Resize(self.output_size)
        image = tr(image)

        return {'image': image, 'label': sample['label']}


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, int)
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        tr = torchvision.transforms.RandomCrop(self.output_size)
        image = tr(image)

        return {'image': image, 'label': label}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        tr = torchvision.transforms.ToTensor()
        image = tr(image)

        label = torch.tensor(int(label))

        return {'image': image,
                'label': label}


COMPOSED_TRANSFORM = Compose([
        ToTensor(),
        Rescale(RESCALE_TO),
        # RandomCrop(400),
        Normalize(DATA_MEAN, DATA_STD)
    ])
