from torch.utils.data import Dataset, DataLoader
from tensorflow import keras
from utils.utils import *
from prettytable import PrettyTable


class Fashion_MNIST_DATASET(Dataset):
    def __init__(self, X, Y, ids, transform=None):
        self.X = X
        self.Y = Y
        self.ids = ids
        self.transform = transform

    def __getitem__(self, item):
        x, y = self.X[self.ids[item]], self.Y[self.ids[item]]
        assert x.shape == (28, 28)
        if self.transform is not None:
            x = self.transform(x)
        else:
            x = torch.tensor(x).float()
        y = torch.tensor(y).long()
        return x, y

    def __len__(self):
        return len(self.ids)


def _get_fashion_mnist_dataloaders(X, Y, shard_per_client=2, num_users=100, batch_size=10, transform=None, novel_clients=None, training=False):
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}

    idx_dict = {}
    for i in range(len(X)):
        label = Y[i]
        if label not in idx_dict.keys():
            idx_dict[label] = []
        idx_dict[label].append(i)

    num_classes = 10
    shard_per_class = int(shard_per_client * num_users / num_classes)  # 2 shards per client * 100 clients / 10 classes
    for label in idx_dict.keys():
        x = idx_dict[label]
        num_leftover = len(x) % shard_per_class
        leftover = x[-num_leftover:] if num_leftover > 0 else []
        x = np.array(x[:-num_leftover]) if num_leftover > 0 else np.array(x)
        x = x.reshape((shard_per_class, -1))
        x = list(x)

        for i, idx in enumerate(leftover):
            x[i] = np.concatenate([x[i], [idx]])
        idx_dict[label] = x

    rand_set_all = list(range(num_classes)) * shard_per_class
    random.shuffle(rand_set_all)
    rand_set_all = np.array(rand_set_all).reshape((num_users, -1))

    # divide and assign
    for i in range(num_users):
        rand_set_label = rand_set_all[i]
        rand_set = []
        for label in rand_set_label:
            idx = np.random.choice(len(idx_dict[label]), replace=False)
            rand_set.append(idx_dict[label].pop(idx))
        dict_users[i] = np.concatenate(rand_set)

    test = []
    for key, value in dict_users.items():
        x = np.unique(Y[value])
        assert len(x) <= shard_per_client
        test.append(value)
    test = np.concatenate(test)
    assert len(test) == len(X)
    assert len(set(list(test))) == len(X)

    train_dataloaders = {}
    validation_dataloaders = {}
    test_dataloaders = {}
    for user_id, ids in dict_users.items():
        if user_id < num_users - novel_clients:  # training clients
            # Train and validation split ratio: 85:15
            random.shuffle(dict_users[user_id])
            train_ids = dict_users[user_id][:int(0.85 * len(dict_users[user_id]))]
            train_dataset = Fashion_MNIST_DATASET(X, Y, ids=train_ids, transform=transform)
            train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
            train_dataloaders[user_id] = train_dataloader
            validation_ids = dict_users[user_id][int(0.85 * len(dict_users[user_id])):]
            validation_dataset = Fashion_MNIST_DATASET(X, Y, ids=validation_ids, transform=transform)
            validation_dataloader = DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=True,
                                               num_workers=0)
            validation_dataloaders[user_id] = validation_dataloader
            test_dataloaders[user_id] = None
        else:
            train_dataloaders[user_id] = None
            validation_dataloaders[user_id] = None
            test_dataset = Fashion_MNIST_DATASET(X, Y, ids=dict_users[user_id], transform=transform)
            test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

            test_dataloaders[user_id] = test_dataloader
    users = [i for i in range(num_users)]
    return users, train_dataloaders, validation_dataloaders, test_dataloaders


def get_fashion_mnist_dataloaders(num_users=100, batch_size=10, shard_per_client=2, transform=None, novel_clients=10):
    setup_seed(24)
    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
    y_train = y_train.reshape((60000,))
    y_test = y_test.reshape((10000,))
    x = np.concatenate((x_train, x_test), axis=0)
    y = np.concatenate((y_train, y_test), axis=0)

    table = PrettyTable(['TrainingX.', 'TrainingY.', 'TestX.', 'TestY.'])
    table.add_row([x_train.shape, y_train.shape, x_test.shape, y_test.shape])
    print(table)

    clients, train_loaders, validation_loaders, test_loaders = _get_fashion_mnist_dataloaders(X=x, Y=y,
                                                                                      shard_per_client=shard_per_client,
                                                                                      num_users=num_users,
                                                                                      batch_size=batch_size,
                                                                                      transform=transform,
                                                                                      novel_clients=novel_clients)

    return clients, train_loaders, validation_loaders, test_loaders


if __name__ == '__main__':
    clients, train_loaders, validation_loaders, test_loaders = get_fashion_mnist_dataloaders(num_users=100,
                                                                                             batch_size=64,
                                                                                             shard_per_client=2,
                                                                                             transform=None,
                                                                                             novel_clients=50)
    print(clients)
    for c, d in train_loaders.items():
        for step, (x, y) in enumerate(d):
            print(y)
            break
        # exit(0)