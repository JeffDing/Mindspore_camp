from mindtorch.tools import mstorch_enable
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms.functional import InterpolationMode
import mindspore as ms
import argparse
import time

### data prepare
transform = transforms.Compose([transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.2435, 0.2616])
                               ])


### model construct
class AlexNet(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super(AlexNet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, (11, 11), (4, 4), (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((3, 3), (2, 2)),

            nn.Conv2d(64, 192, (5, 5), (1, 1), (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((3, 3), (2, 2)),

            nn.Conv2d(192, 384, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(),
            nn.Conv2d(384, 256, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(),
            nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(),
            nn.MaxPool2d((3, 3), (2, 2)),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        return self._forward_impl(x)

    # Support torch.script function
    def _forward_impl(self, x):
        out = self.features(x)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out

criterion = nn.CrossEntropyLoss()


################# model train ###############
def train(config_args):
    train_images = datasets.CIFAR10(config_args.dataset, train=True, download=True, transform=transform)
    train_data = DataLoader(train_images, batch_size=16, shuffle=True, num_workers=0, drop_last=True)

    epochs = config_args.epoch
    net = AlexNet().to(config_args.device)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)

    def forward_fn(data, label):
        logits = net(data)
        loss = criterion(logits, label)
        return loss, logits

    grad_fn = ms.ops.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)

    def train_net(data, label):
        (loss, _), grads = grad_fn(data, label)
        optimizer(grads)
        return loss

    net.train()
    print("begin training ......")
    for i in range(epochs):
        epoch_begin = time.time()
        for X, y in train_data:
            res = train_net(X, y)
            print("---------------------->epoch:{}, loss:{:.6f}".format(i, res.asnumpy()))
        print("--------------->epoch:{}, total time:{:.6f}".format(i, time.time() - epoch_begin))
    torch.save(net.state_dict(), config_args.save_path)


################# model eval ###############
def test(config_args):
    test_images = datasets.CIFAR10(config_args.dataset, train=False, download=True, transform=transform)
    test_data = DataLoader(test_images, batch_size=128, shuffle=True, num_workers=4, drop_last=True)

    net = AlexNet().to(config_args.device)
    net.load_state_dict(torch.load(config_args.load_path), strict=True)
    size = len(test_data.dataset)
    num_batches = len(test_data)
    net.eval()
    test_loss, correct = 0, 0
    print("begin testing ......")
    with torch.no_grad():   # comment out this line for graph mode accelerating
        for X, y in test_data:
            X, y = X.to(config_args.device), y.to(config_args.device)
            pred = net(X)
            test_loss += criterion(pred, y).item()
            correct += (pred.argmax(1) == y).to(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


if __name__ == "__main__":
    seed = 1
    torch.manual_seed(seed)

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', help='Execute training or testing.')
    parser.add_argument('--device', type=str, default='Ascend', help='Select the hardware device for execution.')
    parser.add_argument('--epoch', type=int, default=20, help='Epoch size of training.')
    parser.add_argument('--save_path', type=str, default='./alexnet.pth', help='Training output path for local.')
    parser.add_argument('--load_path', type=str, default='./alexnet.pth',
                        help='Pretrained checkpoint path for fine tune or evaluating.')
    parser.add_argument('--dataset', default='./', help='Dataset root directory path')
    config_args = parser.parse_args()


    if config_args.device in ("gpu", "GPU", "cuda"):
        ms.context.set_context(device_target="GPU")
    elif config_args.device in ("cpu", "CPU"):
        ms.context.set_context(device_target="CPU")
    elif config_args.device == "Ascend":
        ms.context.set_context(device_target="Ascend")
    else:
        print("WARNING: '--device' configuration is abnormal, and the appropriate device will be adapted.")

    # for graph mode accelerating
    # ms.context.set_context(mode=ms.GRAPH_MODE)
    # ms.set_context(jit_syntax_level=ms.STRICT)

    if config_args.mode == 'train':
        train(config_args)
    elif config_args.mode == 'test':
        test(config_args)