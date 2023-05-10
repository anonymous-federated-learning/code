from torch.utils.data import Dataset, DataLoader
from tensorflow import keras
from utils.utils import *
from prettytable import PrettyTable


class CIFAR10_DATASET(Dataset):
    def __init__(self, X, Y, ids, transform=None):
        self.X = X
        self.Y = Y
        self.ids = ids
        self.transform = transform

    def __getitem__(self, item):
        x, y = self.X[self.ids[item]], self.Y[self.ids[item]]
        assert x.shape == (32, 32, 3)
        if self.transform is not None:
            x = self.transform(x)
        else:
            x = torch.tensor(x).float()
        y = torch.tensor(y).long()
        return x, y

    def __len__(self):
        return len(self.ids)


def partition_data(x, y, num_users=50, alpha=0.5):
    clients_ids = {i: [] for i in range(num_users)}
    for k in range(10):
        ids = np.where(y == k)[0]
        np.random.shuffle(ids)
        proportions = np.random.dirichlet(np.repeat(alpha, num_users))
        batch_sizes = [int(p * len(ids)) for p in proportions]
        start = 0
        for i in range(num_users):
            size = batch_sizes[i]
            clients_ids[i] += ids[start: start + size].tolist()
            start += size
    return clients_ids


def _get_cifar10_dirichlet_dataloaders(X_train, Y_train, X_test, Y_test, train_alpha=0.1, novel_alpha=0.5, training_clients=100, novel_clients=50, batch_size=10, transform=None):
    train_clients_ids = partition_data(X_train, Y_train, num_users=training_clients, alpha=train_alpha)
    test_clients_ids = partition_data(X_test, Y_test, num_users=novel_clients, alpha=novel_alpha)
    all_clients_ids = {}
    for i in range(training_clients+novel_clients):
        if i < training_clients:
            all_clients_ids.update({i: train_clients_ids[i]})
        else:
            all_clients_ids.update({i: test_clients_ids[i - training_clients]})
    print(all_clients_ids)
    train_dataloaders = {}
    validation_dataloaders = {}
    test_dataloaders = {}

    for user_id, ids in all_clients_ids.items():
        if user_id < training_clients:
            # Train and validation split ratio: 85:15
            random.shuffle(ids)
            train_ids = ids[:int(0.85 * len(ids))]
            train_dataset = CIFAR10_DATASET(X_train, Y_train, ids=train_ids, transform=transform)
            train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
            train_dataloaders[user_id] = train_dataloader

            validation_ids = ids[int(0.85 * len(ids)):]
            validation_dataset = CIFAR10_DATASET(X_train, Y_train, ids=validation_ids, transform=transform)
            validation_dataloader = DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
            validation_dataloaders[user_id] = validation_dataloader
            test_dataloaders[user_id] = None
        else:
            train_dataloaders[user_id] = None
            validation_dataloaders[user_id] = None
            if len(ids) <= 0:
                continue
            test_dataset = CIFAR10_DATASET(X_test, Y_test, ids=ids, transform=transform)
            test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
            test_dataloaders[user_id] = test_dataloader
    train_users = [k for k, v in train_dataloaders.items() if v is not None]
    novel_users = [k for k, v in test_dataloaders.items() if v is not None]
    users = train_users + novel_users
    return users, train_dataloaders, validation_dataloaders, test_dataloaders


def get_cifar10_dirichlet_dataloaders(training_clients=50, novel_clients=50, train_alpha=0.1, novel_alpha=0.5, batch_size=10, transform=None):
    setup_seed(24)
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    y_train = y_train.reshape((50000,))
    y_test = y_test.reshape((10000,))
    # x = np.concatenate((x_train, x_test), axis=0)
    # y = np.concatenate((y_train, y_test), axis=0)

    table = PrettyTable(['TrainingX.', 'TrainingY.', 'TestX.', 'TestY.'])
    table.add_row([x_train.shape, y_train.shape, x_test.shape, y_test.shape])
    print(table)

    clients, train_loaders, validation_loaders, test_loaders = _get_cifar10_dirichlet_dataloaders(X_train=x_train,
                                                                                                  Y_train=y_train,
                                                                                                  X_test=x_test,
                                                                                                  Y_test=y_test,
                                                                                                  training_clients=training_clients,
                                                                                                  novel_clients=novel_clients,
                                                                                                  train_alpha=train_alpha,
                                                                                                  novel_alpha=novel_alpha,
                                                                                                  batch_size=batch_size,
                                                                                                  transform=transform)
    return clients, train_loaders, validation_loaders, test_loaders


if __name__ == '__main__':
    clients, train_loaders, validation_loaders, test_loaders = get_cifar10_dirichlet_dataloaders(novel_alpha=0.1)
