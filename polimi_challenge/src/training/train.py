import torch.nn as nn
import torch
import matplotlib.pyplot as plt

from polimi_challenge.src.params import DEVICE


def overfit_batch(net: nn.Module, batch: torch.Tensor, iterations=500):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=1e-4, momentum=0.8)
    softmax = torch.nn.Softmax(dim=0)

    images = batch["image"].to(DEVICE)
    labels = batch["label"].to(DEVICE)

    lines_0 = [
        [], [], []
    ]

    lines_1 = [
        [], [], []
    ]

    lines_2 = [
        [], [], []
    ]

    lines_3 = [
        [], [], []
    ]

    for i in range(iterations):

        optimizer.zero_grad()

        out = net(images)

        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()

        out_prob = softmax(out[0])
        for index, el in enumerate(out_prob):
            lines_0[index].append(el.item())

        out_prob = softmax(out[1])
        for index, el in enumerate(out_prob):
            lines_1[index].append(el.item())

        out_prob = softmax(out[2])
        for index, el in enumerate(out_prob):
            lines_2[index].append(el.item())

        out_prob = softmax(out[3])
        for index, el in enumerate(out_prob):
            lines_3[index].append(el.item())

        if i % int(iterations / 10) == 0:
            print("Iteration {} of {}".format(i, iterations))

    x = [i for i in range(len(lines_0[0]))]

    plt.plot(x, lines_0[0], label="Class 0")
    plt.plot(x, lines_0[1], label="Class 1")
    plt.plot(x, lines_0[2], label="Class 2")
    plt.title("Should predict class {}".format(labels[0].item()))
    plt.legend()
    plt.show()

    plt.plot(x, lines_1[0], label="Class 0")
    plt.plot(x, lines_1[1], label="Class 1")
    plt.plot(x, lines_1[2], label="Class 2")
    plt.title("Should predict class {}".format(labels[1].item()))
    plt.legend()
    plt.show()

    plt.plot(x, lines_2[0], label="Class 0")
    plt.plot(x, lines_2[1], label="Class 1")
    plt.plot(x, lines_2[2], label="Class 2")
    plt.title("Should predict class {}".format(labels[2].item()))
    plt.legend()
    plt.show()

    plt.plot(x, lines_3[0], label="Class 0")
    plt.plot(x, lines_3[1], label="Class 1")
    plt.plot(x, lines_3[2], label="Class 2")
    plt.title("Should predict class {}".format(labels[3].item()))
    plt.legend()
    plt.show()

    final_prediction = net(images)
    for index, label in enumerate(labels):
        _, predicted = torch.max(final_prediction, 1)
        print("Predicion with label {}: {}".format(label.item(), predicted[index]))

