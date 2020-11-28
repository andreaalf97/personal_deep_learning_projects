import torch.nn as nn


class Resnet(nn.Module):

    def __init__(self):
        super(Resnet, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 9, 36)
            # nn.Conv2d(9, 18, 9)
            # nn.BatchNorm2d(9)
        )

    def forward(self, x):
        x = self.conv1(x)

        return x


class Net(nn.Module):

    def __init__(self, use_bn=False):
        super(Net, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 6, 100),
            nn.AvgPool2d(4, 4),
            nn.ReLU(),

            nn.Conv2d(6, 6, 5),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.flattened_dimension = 6 * 35 * 35
        self.dropout_prob = 0.2

        self.linear = nn.Sequential(
            nn.Linear(self.flattened_dimension, 100),
            nn.Dropout(p=self.dropout_prob),
            nn.ReLU(),
            nn.Linear(100, 15),
            nn.Dropout(p=self.dropout_prob),
            nn.ReLU(),
            nn.Linear(15, 3)
        )

    def forward(self, x):
        #         print("Input shape: {}".format(x.shape))

        x = self.conv(x)

        # print("Shape after CONVOLUTION: {}".format(x.shape))

        x = x.view(-1, self.flattened_dimension)

        x = self.linear(x)
        #         print("Shape after all linear layers: {}".format(x.shape))

        return x