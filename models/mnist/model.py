import torch
from torch import nn


class MNIST_LeNet5(nn.Module):
    def __init__(self):
        super(MNIST_LeNet5, self).__init__()
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
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.clf(x)
        return x


# class MNIST_2NN(nn.Module):
#     def __init__(self):
#         super(MNIST_2NN, self).__init__()
#         self.clf = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(28 * 28, 200),
#             nn.ReLU(inplace=True),
#             nn.Linear(200, 200),
#             nn.ReLU(inplace=True),
#             nn.Linear(200, 10)
#         )
#
#     def forward(self, x):
#         x = self.clf(x)
#         return x

class MNIST_2NN(nn.Module):
    def __init__(self, config):
        super(MNIST_2NN, self).__init__()
        if config.algorithm == 'fedsr':
            self.conv = nn.Sequential(
                nn.Flatten(),
                nn.Linear(28 * 28, 200),
                nn.ReLU(inplace=True),
                nn.Linear(200, 400),
                nn.ReLU(inplace=True),
            )
        else:
            self.conv = nn.Sequential(
                nn.Flatten(),
                nn.Linear(28 * 28, 200),
                nn.ReLU(inplace=True),
                nn.Linear(200, 200),
                nn.ReLU(inplace=True),
            )

        self.clf = nn.Sequential(
            nn.Linear(200, 10)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.clf(x)
        return x

if __name__ == '__main__':
    model = MNIST_2NN()
    x = torch.ones((50, 1, 28, 28))
    out = model(x)
    print(out.shape)
    num_params = sum(param.numel() for param in model.parameters())
    print(num_params)