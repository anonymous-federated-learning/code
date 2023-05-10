import torch
import torch.nn as nn


class MNIST_Encoder(nn.Module):
    def __init__(self):
        super(MNIST_Encoder, self).__init__()
        self.clf = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 200),
            nn.ReLU(inplace=True),
            nn.Linear(200, 32)
        )

    def forward(self, x):
        x = self.clf(x)
        x = torch.mean(x, dim=0)
        return x


class CIFAR10_Encoder(nn.Module):  # 2156490
    def __init__(self):
        super(CIFAR10_Encoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(5, 5), padding=(2, 2)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 5), padding=(2, 2)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Flatten()
        )
        self.clf = nn.Sequential(
            nn.Linear(64 * 8 * 8, 32)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.clf(x)
        x = torch.mean(x, dim=0)
        return x


class FEMNIST_Encoder(nn.Module):
    def __init__(self):
        super(FEMNIST_Encoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(5, 5), padding=(2, 2)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 5), padding=(2, 2)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Flatten()
        )
        self.clf = nn.Sequential(
            nn.Linear(64 * 7 * 7, 32)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.clf(x)
        x = torch.mean(x, dim=0)
        return x


class FashionMNIST_Encoder(nn.Module):
    def __init__(self):
        super(FashionMNIST_Encoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(5, 5), padding=(2, 2)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 5), padding=(2, 2)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Flatten()
        )
        self.clf = nn.Sequential(
            nn.Linear(64 * 7 * 7, 32)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.clf(x)
        x = torch.mean(x, dim=0)
        return x

class CIFAR100_Encoder(nn.Module):  # 2156490
    def __init__(self):
        super(CIFAR100_Encoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(5, 5), padding=(2, 2)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 5), padding=(2, 2)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Flatten()
        )
        self.clf = nn.Sequential(
            nn.Linear(64 * 8 * 8, 32)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.clf(x)
        x = torch.mean(x, dim=0)
        return x

if __name__ == '__main__':
    # model = MNIST_Encoder()
    # x = torch.ones((50, 1, 28, 28))
    model = CIFAR10_Encoder()
    x = torch.ones((50, 3, 32, 32))
    out = model(x)
    print(out.shape)
    num_params = sum(param.numel() for param in model.parameters())
    print(num_params)