import torch
from torch import nn


class FEMNIST_LeNet5(nn.Module):
    def __init__(self, config):
        super(FEMNIST_LeNet5, self).__init__()
        if config.algorithm == 'fedsr':
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(5, 5), padding=(2, 2)),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(in_channels=32, out_channels=128, kernel_size=(5, 5), padding=(2, 2)),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                nn.Flatten()
            )
        else:
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
            nn.Linear(64 * 7 * 7, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 62)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.clf(x)
        return x


if __name__ == '__main__':
    model = FEMNIST_LeNet5()
    x = torch.ones((50, 1, 28, 28))
    out = model(x)
    print(out.shape)
    num_params = sum(param.numel() for param in model.parameters())
    print(num_params)

    for k,v in model.named_parameters():
        print(k, v.shape)