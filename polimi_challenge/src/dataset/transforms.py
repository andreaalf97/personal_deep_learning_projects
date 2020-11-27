import torchvision
import torch

DATA_MEAN = (0.5748, 0.5188, 0.4885)
DATA_STD = (0.2855, 0.2774, 0.2793)


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, int)
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

        #         if int(label) == 0:
        #             label = [1, 0, 0]
        #         elif int(label) == 1:
        #             label = [0, 1, 0]
        #         else:
        #             label = [0, 0, 1]

        #         label = torch.tensor(label)
        #         label = label.view(1, -1)

        label = torch.tensor(int(label))

        norm = torchvision.transforms.Normalize(DATA_MEAN, DATA_STD)
        image = norm(image)

        return {'image': image,
                'label': label}
