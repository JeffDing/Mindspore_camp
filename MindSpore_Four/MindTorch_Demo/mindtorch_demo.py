from mindtorch.tools import mstorch_enable
from lenet import LeNet5
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets.mnist import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time

data_train = MNIST('./data/mnist',
                   download=True,
                   transform=transforms.Compose([
                       transforms.Resize((32, 32)),
                       transforms.ToTensor()]))
data_test = MNIST('./data/mnist',
                  train=False,
                  download=True,
                  transform=transforms.Compose([
                      transforms.Resize((32, 32)),
                      transforms.ToTensor()]))
data_train_loader = DataLoader(data_train, batch_size=128, shuffle=True, num_workers=4, drop_last=True)
data_test_loader = DataLoader(data_test, batch_size=128, num_workers=4, drop_last=True)

net = LeNet5()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=2e-3)

cur_batch_win = None
cur_batch_win_opts = {
    'title': 'Epoch Loss Trace',
    'xlabel': 'Batch Number',
    'ylabel': 'Loss',
    'width': 1200,
    'height': 600,
}


def train(epoch):
    global cur_batch_win
    net.train()
    loss_list, batch_list = [], []
    size = len(data_train_loader)
    start = time.time()
    for i, (images, labels) in enumerate(data_train_loader):
        optimizer.zero_grad()

        output = net(images)

        loss = criterion(output, labels)

        loss_list.append(loss.detach().cpu().item())
        batch_list.append(i + 1)

        if i % 10 == 0:
            end = time.time()
            print(f"loss: {loss.detach().cpu().item():>7f}  [{i:>3d}/{size:>3d}]", "Runing time:", end - start, "s")
            start = time.time()
        loss.backward()
        optimizer.step()


def test():
    net.eval()
    total_correct = 0
    avg_loss = 0.0
    num_batches = 0
    total = 0
    for i, (images, labels) in enumerate(data_test_loader):
        output = net(images)
        num_batches += 1
        total += len(images)
        avg_loss += criterion(output, labels).sum()
        pred = output.detach().max(1)[1]
        total_correct += pred.eq(labels.view_as(pred)).sum()

    avg_loss = avg_loss / num_batches
    total_correct = total_correct / total

    print(f"Test: \n Accuracy: {(100 * total_correct):>0.1f}%, Avg loss: {avg_loss.detach().cpu().item():>8f} \n")


def train_and_test(epoch):
    print(f"Epoch {epoch}\n-------------------------------")
    train(epoch)
    test()


def main():
    for e in range(1, 2):
        train_and_test(e)


if __name__ == '__main__':
    main()
