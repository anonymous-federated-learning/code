import torch
import torch.nn as nn


class CIFAR100_LeNet5(nn.Module):  # 2202660
    def __init__(self, config):
        super(CIFAR100_LeNet5, self).__init__()
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
            nn.Linear(512, 100)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.clf(x)
        return x


if __name__ == '__main__':
    model = CIFAR100_LeNet5()
    x = torch.ones((50, 3, 32, 32))
    output = model(x)
    print()
    print(output.shape)
    num_params = sum(param.numel() for param in model.parameters())
    print(num_params)

