from torchvision.transforms import transforms

# datasets
from data.cifar10.cifar10_dataset import get_cifar10_dataloaders
from data.cifar10.cifar10_dataset import get_hetero_cifar10_dataloaders
from data.cifar10.cifar10_diri_dataset import get_cifar10_dirichlet_dataloaders
from data.mnist.mnist_dataset import get_mnist_dataloaders
from data.mnist.mnist_dataset import get_rotated_mnist_dataloaders
from data.cifar100.cifar100_dataset import get_cifar100_dataloaders
from data.femnist.femnist_dataset import get_femnist_dataloaders
from data.fashion_mnist.fashion_mnist import get_fashion_mnist_dataloaders

# models
from models.cifar10.model import CIFAR10_LeNet5
from models.mnist.model import MNIST_LeNet5, MNIST_2NN
from models.cifar100.model import CIFAR100_LeNet5
from models.femnist.model import FEMNIST_LeNet5
from models.fashion_mnist.model import Fashion_MNIST_LeNet5
from models.loss_nn import MLP

from models.cifar10.model import CIFAR10_LeNet5_small
from models.cifar10.model import CIFAR10_LeNet5_medium
from models.cifar10.model import CIFAR10_LeNet5_big

# client encoder
from models.encoder_nn import MNIST_Encoder, CIFAR10_Encoder, FEMNIST_Encoder, FashionMNIST_Encoder, CIFAR100_Encoder

# client hpnet
from models.hyper_net import MNIST_Hyper, CIFAR10_Hyper, FEMNIST_Hyper, FashionMNIST_Hyper, CIFAR100_Hyper


def setup_datasets(dataset, batch_size, num_users=100, classes_per_client=2, novel_clients=10,
                   training_clients=50, train_alpha=0.1, novel_alpha=0.5, ratio_of_test=1.0):
    users, train_loaders, validation_loaders, test_loaders = [], [], [], []
    if dataset == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        users, train_loaders, validation_loaders, test_loaders = get_cifar10_dataloaders(num_users=num_users,
                                                                                         batch_size=batch_size,
                                                                                         transform=transform,
                                                                                         shard_per_client=classes_per_client,
                                                                                         novel_clients=novel_clients,
                                                                                         ratio_of_test=ratio_of_test)
    elif dataset == 'cifar10_diri':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        users, train_loaders, validation_loaders, test_loaders = get_cifar10_dirichlet_dataloaders(training_clients=training_clients,
                                                                                                   novel_clients=novel_clients,
                                                                                                   train_alpha=train_alpha,
                                                                                                   novel_alpha=novel_alpha,
                                                                                                    batch_size=batch_size,
                                                                                                    transform=transform)
    elif dataset == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        users, train_loaders, validation_loaders, test_loaders = get_mnist_dataloaders(num_users=num_users,
                                                                                       batch_size=batch_size,
                                                                                       transform=transform,
                                                                                       shard_per_client=classes_per_client,
                                                                                       novel_clients=novel_clients)
    elif dataset == 'rotated_mnist':
        users, train_loaders, validation_loaders, test_loaders = get_rotated_mnist_dataloaders(num_users=num_users,
                                                                                       batch_size=batch_size,
                                                                                       shard_per_client=classes_per_client,
                                                                                       novel_clients=novel_clients)
    elif dataset == 'cifar100':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                 std=[0.2675, 0.2565, 0.2761])
        ])
        users, train_loaders, validation_loaders, test_loaders = get_cifar100_dataloaders(num_users=num_users,
                                                                                          batch_size=batch_size,
                                                                                          transform=transform,
                                                                                          novel_clients=novel_clients)
    elif dataset == 'femnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        users, train_loaders, validation_loaders, test_loaders = get_femnist_dataloaders(batch_size=batch_size,
                                                                                         transform=transform,
                                                                                         novel_clients=novel_clients)

    elif dataset == 'fashion_mnist':
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        users, train_loaders, validation_loaders, test_loaders = get_fashion_mnist_dataloaders(num_users=num_users,
                                                                                       batch_size=batch_size,
                                                                                       transform=transform,
                                                                                       shard_per_client=classes_per_client,
                                                                                       novel_clients=novel_clients)
    return users, train_loaders, validation_loaders, test_loaders


def setup_hetero_datasets(dataset, batch_size, num_users=100, classes_per_client=2, novel_clients=10, client_train_ratio=0.85, client_val_ratio=0.15):
    users, train_loaders, validation_loaders, test_loaders, public_dataloader = [], [], [], [], None
    if dataset == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        users, train_loaders, validation_loaders, test_loaders, public_dataloader = get_hetero_cifar10_dataloaders(num_users=num_users,
                                                                                         batch_size=batch_size,
                                                                                         transform=transform,
                                                                                         shard_per_client=classes_per_client,
                                                                                         novel_clients=novel_clients,
                                                                                         client_train_ratio=client_train_ratio,
                                                                                         client_val_ratio=client_val_ratio)
    return users, train_loaders, validation_loaders, test_loaders, public_dataloader

def select_model(config):
    model_name = config.model
    model = None
    if model_name == 'cifar10':
        model = CIFAR10_LeNet5(config=config)
    elif model_name == 'mnist':
        model = MNIST_LeNet5()
    elif model_name == 'mnist-2nn':
        model = MNIST_2NN(config=config)
    elif model_name == 'cifar100':
        model = CIFAR100_LeNet5(config=config)
    elif model_name == 'femnist':
        model = FEMNIST_LeNet5(config=config)
    elif model_name == 'fashion_mnist':
        model = Fashion_MNIST_LeNet5(config=config)
    else:
        assert "Unimplemented model!"
    return model

def select_hetero_model(config, model_type):
    model = None
    if config.dataset == 'cifar10':
        if model_type == 'small':
            model = CIFAR10_LeNet5_small()
        elif model_type == 'medium':
            model = CIFAR10_LeNet5_medium()
        elif model_type == 'big':
            model = CIFAR10_LeNet5_big()
    return model

def get_loss_nn(num_classes=10):
    return MLP(in_size=num_classes, norm_reduce=True)

def get_hetero_loss_nn(num_classes=10, model_type='small'):
    loss_nn = None
    if model_type == 'small':
        loss_nn = MLP(in_size=num_classes, hidden_dim=32, norm_reduce=True)
    elif model_type == 'medium':
        loss_nn = MLP(in_size=num_classes, hidden_dim=64, norm_reduce=True)
    elif model_type == 'big':
        loss_nn = MLP(in_size=num_classes, hidden_dim=128, norm_reduce=True)
    return loss_nn

def get_encoder_nn(dataset='mnist'):
    encoder_nn = None
    if dataset in ['cifar10', 'cifar10_diri']:
        encoder_nn = CIFAR10_Encoder()
    elif dataset in ['mnist', 'rotated_mnist']:
        encoder_nn = MNIST_Encoder()
    elif dataset == 'femnist':
        encoder_nn = FEMNIST_Encoder()
    elif dataset == 'fashion_mnist':
        encoder_nn = FashionMNIST_Encoder()
    elif dataset == 'cifar100':
        encoder_nn = CIFAR100_Encoder()
    else:
        assert "Unimplemented model!"
    return encoder_nn


def get_hpnet(dataset='mnist'):
    hpnet = None
    if dataset in ['cifar10', 'cifar10_diri']:
        hpnet = CIFAR10_Hyper()
    elif dataset in ['mnist', 'rotated_mnist']:
        hpnet = MNIST_Hyper()
    elif dataset == 'femnist':
        hpnet = FEMNIST_Hyper()
    elif dataset == 'fashion_mnist':
        hpnet = FashionMNIST_Hyper()
    elif dataset == 'cifar100':
        hpnet = CIFAR100_Hyper()
    else:
        assert "Unimplemented model!"
    return hpnet
