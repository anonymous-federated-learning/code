import torch
import torch.nn as nn


class CIFAR10_LeNet5(nn.Module):  # 2156490
    def __init__(self, config):
        super(CIFAR10_LeNet5, self).__init__()
        if config.algorithm == 'fedsr':
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(5, 5), padding=(2, 2)),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(in_channels=32, out_channels=128, kernel_size=(5, 5), padding=(2, 2)),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                nn.Flatten()
            )
        else:
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
            nn.Linear(64 * 8 * 8, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.clf(x)
        return x


class CIFAR10_LeNet5_small(nn.Module):  # 2156490
    def __init__(self):
        super(CIFAR10_LeNet5_small, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(5, 5), padding=(2, 2)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(5, 5), padding=(2, 2)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Flatten()
        )

        self.clf = nn.Sequential(
            nn.Linear(32 * 8 * 8, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.clf(x)
        return x

class CIFAR10_LeNet5_medium(nn.Module):  # 2156490
    def __init__(self):
        super(CIFAR10_LeNet5_medium, self).__init__()
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
            nn.Linear(64 * 8 * 8, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.clf(x)
        return x

class CIFAR10_LeNet5_big(nn.Module):  # 2156490
    def __init__(self):
        super(CIFAR10_LeNet5_big, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(5, 5), padding=(2, 2)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(5, 5), padding=(2, 2)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Flatten()
        )

        self.clf = nn.Sequential(
            nn.Linear(128 * 8 * 8, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.clf(x)
        return x

if __name__ == '__main__':
    model = CIFAR10_LeNet5()
    x = torch.ones((50, 3, 32, 32))
    output = model(x)
    print()
    print(output.shape)
    num_params = sum(param.numel() for param in model.parameters())
    print(num_params)
    for key, param in model.named_parameters():
        print(key, param.shape)

