from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import spectral_norm


class MNIST_Hyper(nn.Module):
    def __init__(
            self, embedding_dim=32, in_channels=1, out_dim=10, n_kernels=32, hidden_dim=100,
            spec_norm=False, n_hidden=3):
        super().__init__()

        self.in_channels = in_channels
        self.out_dim = out_dim
        self.n_kernels = n_kernels

        layers = [
            spectral_norm(nn.Linear(embedding_dim, hidden_dim)) if spec_norm else nn.Linear(embedding_dim, hidden_dim),
        ]
        for _ in range(n_hidden):
            layers.append(nn.ReLU(inplace=True))
            layers.append(
                spectral_norm(nn.Linear(hidden_dim, hidden_dim)) if spec_norm else nn.Linear(hidden_dim, hidden_dim),
            )

        self.mlp = nn.Sequential(*layers)

        self.clf1_weights = nn.Linear(hidden_dim, 200 * 784)
        self.clf1_bias = nn.Linear(hidden_dim, 200)
        self.clf3_weights = nn.Linear(hidden_dim, 200 * 200)
        self.clf3_bias = nn.Linear(hidden_dim, 200)
        self.clf5_weights = nn.Linear(hidden_dim, 10 * 200)
        self.clf5_bias = nn.Linear(hidden_dim, 10)

        if spec_norm:
            self.clf1_weights = spectral_norm(self.clf1_weights)
            self.clf1_bias = spectral_norm(self.clf1_bias)
            self.clf3_weights = spectral_norm(self.clf3_weights)
            self.clf3_bias = spectral_norm(self.clf3_bias)
            self.clf5_weights = spectral_norm(self.clf5_weights)
            self.clf5_bias = spectral_norm(self.clf5_bias)

    def forward(self, emd):
        features = self.mlp(emd)

        weights = OrderedDict({
            "conv.1.weight": self.clf1_weights(features).view(200, 784),
            "conv.1.bias": self.clf1_bias(features).view(-1),
            "conv.3.weight": self.clf3_weights(features).view(200, 200),
            "conv.3.bias": self.clf3_bias(features).view(-1),
            "clf.0.weight": self.clf5_weights(features).view(10, 200),
            "clf.0.bias": self.clf5_bias(features).view(-1),
        })
        return weights


class CIFAR10_Hyper(nn.Module):
    def __init__(
            self, embedding_dim=32, in_channels=3, out_dim=10, n_kernels=32, hidden_dim=100,
            spec_norm=False, n_hidden=3):
        super().__init__()

        self.in_channels = in_channels
        self.out_dim = out_dim
        self.n_kernels = n_kernels

        layers = [
            spectral_norm(nn.Linear(embedding_dim, hidden_dim)) if spec_norm else nn.Linear(embedding_dim, hidden_dim),
        ]
        for _ in range(n_hidden):
            layers.append(nn.ReLU(inplace=True))
            layers.append(
                spectral_norm(nn.Linear(hidden_dim, hidden_dim)) if spec_norm else nn.Linear(hidden_dim, hidden_dim),
            )

        self.mlp = nn.Sequential(*layers)

        self.conv0_weights = nn.Linear(hidden_dim, 32 * 3 * 5 * 5)
        self.conv0_bias = nn.Linear(hidden_dim, 32)
        self.conv3_weights = nn.Linear(hidden_dim, 64 * 32 * 5 * 5)
        self.conv3_bias = nn.Linear(hidden_dim, 64)

        self.clf0_weights = nn.Linear(hidden_dim, 512 * 4096)
        self.clf0_bias = nn.Linear(hidden_dim, 512)
        self.clf2_weights = nn.Linear(hidden_dim, 512 * 10)
        self.clf2_bias = nn.Linear(hidden_dim, 10)

        if spec_norm:
            self.conv0_weights = spectral_norm(self.conv0_weights)
            self.conv0_bias = spectral_norm(self.conv0_bias)
            self.conv3_weights = spectral_norm(self.conv3_weights)
            self.conv3_bias = spectral_norm(self.conv3_bias)

            self.clf0_weights = spectral_norm(self.clf0_weights)
            self.clf0_bias = spectral_norm(self.clf0_bias)
            self.clf2_weights = spectral_norm(self.clf2_weights)
            self.clf2_bias = spectral_norm(self.clf2_bias)

    def forward(self, emd):
        features = self.mlp(emd)

        weights = OrderedDict({
            "conv.0.weight": self.conv0_weights(features).view(32, 3, 5, 5),
            "conv.0.bias": self.conv0_bias(features).view(-1),
            "conv.3.weight": self.conv3_weights(features).view(64, 32, 5, 5),
            "conv.3.bias": self.conv3_bias(features).view(-1),
            "clf.0.weight": self.clf0_weights(features).view(512, 4096),
            "clf.0.bias": self.clf0_bias(features).view(-1),
            "clf.2.weight": self.clf2_weights(features).view(10, 512),
            "clf.2.bias": self.clf2_bias(features).view(-1)
        })
        return weights


class FEMNIST_Hyper(nn.Module):
    def __init__(
            self, embedding_dim=32, in_channels=1, out_dim=62, n_kernels=32, hidden_dim=100,
            spec_norm=False, n_hidden=3):
        super().__init__()

        self.in_channels = in_channels
        self.out_dim = out_dim
        self.n_kernels = n_kernels

        layers = [
            spectral_norm(nn.Linear(embedding_dim, hidden_dim)) if spec_norm else nn.Linear(embedding_dim, hidden_dim),
        ]
        for _ in range(n_hidden):
            layers.append(nn.ReLU(inplace=True))
            layers.append(
                spectral_norm(nn.Linear(hidden_dim, hidden_dim)) if spec_norm else nn.Linear(hidden_dim, hidden_dim),
            )

        self.mlp = nn.Sequential(*layers)

        self.conv0_weights = nn.Linear(hidden_dim, 32 * 1 * 5 * 5)
        self.conv0_bias = nn.Linear(hidden_dim, 32)
        self.conv3_weights = nn.Linear(hidden_dim, 64 * 32 * 5 * 5)
        self.conv3_bias = nn.Linear(hidden_dim, 64)

        self.clf0_weights = nn.Linear(hidden_dim, 2048 * 3136)
        self.clf0_bias = nn.Linear(hidden_dim, 2048)
        self.clf2_weights = nn.Linear(hidden_dim, 62 * 2048)
        self.clf2_bias = nn.Linear(hidden_dim, 62)

        if spec_norm:
            self.conv0_weights = spectral_norm(self.conv0_weights)
            self.conv0_bias = spectral_norm(self.conv0_bias)
            self.conv3_weights = spectral_norm(self.conv3_weights)
            self.conv3_bias = spectral_norm(self.conv3_bias)

            self.clf0_weights = spectral_norm(self.clf0_weights)
            self.clf0_bias = spectral_norm(self.clf0_bias)
            self.clf2_weights = spectral_norm(self.clf2_weights)
            self.clf2_bias = spectral_norm(self.clf2_bias)

    def forward(self, emd):
        features = self.mlp(emd)

        weights = OrderedDict({
            "conv.0.weight": self.conv0_weights(features).view(32, 1, 5, 5),
            "conv.0.bias": self.conv0_bias(features).view(-1),
            "conv.3.weight": self.conv3_weights(features).view(64, 32, 5, 5),
            "conv.3.bias": self.conv3_bias(features).view(-1),
            "clf.0.weight": self.clf0_weights(features).view(2048, 3136),
            "clf.0.bias": self.clf0_bias(features).view(-1),
            "clf.2.weight": self.clf2_weights(features).view(62, 2048),
            "clf.2.bias": self.clf2_bias(features).view(-1)
        })
        return weights


class FashionMNIST_Hyper(nn.Module):
    def __init__(
            self, embedding_dim=32, in_channels=1, out_dim=62, n_kernels=32, hidden_dim=100,
            spec_norm=False, n_hidden=3):
        super().__init__()

        self.in_channels = in_channels
        self.out_dim = out_dim
        self.n_kernels = n_kernels

        layers = [
            spectral_norm(nn.Linear(embedding_dim, hidden_dim)) if spec_norm else nn.Linear(embedding_dim, hidden_dim),
        ]
        for _ in range(n_hidden):
            layers.append(nn.ReLU(inplace=True))
            layers.append(
                spectral_norm(nn.Linear(hidden_dim, hidden_dim)) if spec_norm else nn.Linear(hidden_dim, hidden_dim),
            )

        self.mlp = nn.Sequential(*layers)

        self.conv0_weights = nn.Linear(hidden_dim, 32 * 1 * 5 * 5)
        self.conv0_bias = nn.Linear(hidden_dim, 32)
        self.conv3_weights = nn.Linear(hidden_dim, 64 * 32 * 5 * 5)
        self.conv3_bias = nn.Linear(hidden_dim, 64)

        self.clf0_weights = nn.Linear(hidden_dim, 512 * 3136)
        self.clf0_bias = nn.Linear(hidden_dim, 512)
        self.clf2_weights = nn.Linear(hidden_dim, 10 * 512)
        self.clf2_bias = nn.Linear(hidden_dim, 10)

        if spec_norm:
            self.conv0_weights = spectral_norm(self.conv0_weights)
            self.conv0_bias = spectral_norm(self.conv0_bias)
            self.conv3_weights = spectral_norm(self.conv3_weights)
            self.conv3_bias = spectral_norm(self.conv3_bias)

            self.clf0_weights = spectral_norm(self.clf0_weights)
            self.clf0_bias = spectral_norm(self.clf0_bias)
            self.clf2_weights = spectral_norm(self.clf2_weights)
            self.clf2_bias = spectral_norm(self.clf2_bias)

    def forward(self, emd):
        features = self.mlp(emd)

        weights = OrderedDict({
            "conv.0.weight": self.conv0_weights(features).view(32, 1, 5, 5),
            "conv.0.bias": self.conv0_bias(features).view(-1),
            "conv.3.weight": self.conv3_weights(features).view(64, 32, 5, 5),
            "conv.3.bias": self.conv3_bias(features).view(-1),
            "clf.0.weight": self.clf0_weights(features).view(512, 3136),
            "clf.0.bias": self.clf0_bias(features).view(-1),
            "clf.2.weight": self.clf2_weights(features).view(10, 512),
            "clf.2.bias": self.clf2_bias(features).view(-1)
        })
        return weights


class CIFAR100_Hyper(nn.Module):
    def __init__(
            self, embedding_dim=32, in_channels=3, out_dim=10, n_kernels=32, hidden_dim=100,
            spec_norm=False, n_hidden=3):
        super().__init__()

        self.in_channels = in_channels
        self.out_dim = out_dim
        self.n_kernels = n_kernels

        layers = [
            spectral_norm(nn.Linear(embedding_dim, hidden_dim)) if spec_norm else nn.Linear(embedding_dim, hidden_dim),
        ]
        for _ in range(n_hidden):
            layers.append(nn.ReLU(inplace=True))
            layers.append(
                spectral_norm(nn.Linear(hidden_dim, hidden_dim)) if spec_norm else nn.Linear(hidden_dim, hidden_dim),
            )

        self.mlp = nn.Sequential(*layers)

        self.conv0_weights = nn.Linear(hidden_dim, 32 * 3 * 5 * 5)
        self.conv0_bias = nn.Linear(hidden_dim, 32)
        self.conv3_weights = nn.Linear(hidden_dim, 64 * 32 * 5 * 5)
        self.conv3_bias = nn.Linear(hidden_dim, 64)

        self.clf0_weights = nn.Linear(hidden_dim, 512 * 4096)
        self.clf0_bias = nn.Linear(hidden_dim, 512)
        self.clf2_weights = nn.Linear(hidden_dim, 512 * 100)
        self.clf2_bias = nn.Linear(hidden_dim, 100)

        if spec_norm:
            self.conv0_weights = spectral_norm(self.conv0_weights)
            self.conv0_bias = spectral_norm(self.conv0_bias)
            self.conv3_weights = spectral_norm(self.conv3_weights)
            self.conv3_bias = spectral_norm(self.conv3_bias)

            self.clf0_weights = spectral_norm(self.clf0_weights)
            self.clf0_bias = spectral_norm(self.clf0_bias)
            self.clf2_weights = spectral_norm(self.clf2_weights)
            self.clf2_bias = spectral_norm(self.clf2_bias)

    def forward(self, emd):
        features = self.mlp(emd)

        weights = OrderedDict({
            "conv.0.weight": self.conv0_weights(features).view(32, 3, 5, 5),
            "conv.0.bias": self.conv0_bias(features).view(-1),
            "conv.3.weight": self.conv3_weights(features).view(64, 32, 5, 5),
            "conv.3.bias": self.conv3_bias(features).view(-1),
            "clf.0.weight": self.clf0_weights(features).view(512, 4096),
            "clf.0.bias": self.clf0_bias(features).view(-1),
            "clf.2.weight": self.clf2_weights(features).view(100, 512),
            "clf.2.bias": self.clf2_bias(features).view(-1)
        })
        return weights

if __name__ == '__main__':
    hnet = FEMNIST_Hyper(embedding_dim=64, in_channels=3, out_dim=10, n_kernels=32, hidden_dim=100, spec_norm=True, n_hidden=3)
    print(hnet)
    p_weights = hnet(torch.ones((1, 64)))
    for k, v in p_weights.items():
        print(k, v.shape)
