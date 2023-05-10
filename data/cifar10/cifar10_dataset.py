from torch.utils.data import Dataset, DataLoader
from tensorflow import keras
from utils.utils import *
from torchvision import transforms
from torchvision.transforms.functional import rotate
from PIL import Image, ImageFile
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


class SOFT_CIFAR10_DATASET(Dataset):
    def __init__(self, X, Y, pseudo_Y=None, transform=None):
        self.X = X
        self.Y = Y
        self.pseudo_Y = pseudo_Y
        self.transform = transform

    def __getitem__(self, item):
        x, y = self.X[item], self.Y[item]
        assert x.shape == (32, 32, 3)
        pseudo_y = self.pseudo_Y[item] if self.pseudo_Y is not None else -1
        if self.transform is not None:
            x = self.transform(x)
        else:
            x = torch.tensor(x).float()
        y = torch.tensor(y).long()
        return x, y, pseudo_y

    def __len__(self):
        return self.Y.shape[0]


def _get_cifar10_dataloaders(X, Y, num_users=100, batch_size=10, transform=None, novel_clients=None, shard_per_client=2, ratio_of_test=1.0):
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
    shard_left = {i: shard_per_class for i in range(num_classes)}
    for i in range(num_users):
        available_class = [key for (key, value) in shard_left.items() if value > 0]
        try:
            selected_labels = np.random.choice(available_class, shard_per_client, replace=False)
            for l in selected_labels:
                shard_left[l] -= 1
        except:
            selected_labels = []
            while len(selected_labels) < shard_per_client:
                available_class = [key for (key, value) in shard_left.items() if value > 0]
                l = np.random.choice(available_class, 1, replace=True)[0]
                selected_labels.append(l)
                shard_left[l] -= 1
        rand_set = []
        for label in selected_labels:
            idx = np.random.choice(len(idx_dict[label]), replace=False)
            rand_set.append(idx_dict[label].pop(idx))
        dict_users[i] = np.concatenate(rand_set)
    for k, v in shard_left.items():
        assert v == 0
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
        if user_id < num_users - novel_clients:
            # Train and validation split ratio: 85:15
            random.shuffle(ids)
            train_ids = ids[:int(0.85 * len(ids))]
            train_dataset = CIFAR10_DATASET(X, Y, ids=train_ids, transform=transform)
            train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
            train_dataloaders[user_id] = train_dataloader

            validation_ids = ids[int(0.85 * len(ids)):]
            validation_dataset = CIFAR10_DATASET(X, Y, ids=validation_ids, transform=transform)
            validation_dataloader = DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
            validation_dataloaders[user_id] = validation_dataloader
            test_dataloaders[user_id] = None
        else:
            train_dataloaders[user_id] = None
            validation_dataloaders[user_id] = None
            random.shuffle(ids)
            small_size_ids = ids[:int(ratio_of_test * len(ids))]
            test_dataset = CIFAR10_DATASET(X, Y, ids=small_size_ids, transform=transform)
            test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
            test_dataloaders[user_id] = test_dataloader
    users = [i for i in range(num_users)]
    return users, train_dataloaders, validation_dataloaders, test_dataloaders


def _get_rotated_cifar10_dataloaders(X, Y, num_users=100, batch_size=10, novel_clients=None, shard_per_client=2):
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    angles = [0, 15, 30, 45, 60, 75]
    rotation_banks = []
    for angle in angles:
        rotation_banks.append(
            transforms.Compose([
                transforms.ToPILImage(),
                transforms.Lambda(lambda x: rotate(x, angle,
                                                   resample=Image.BICUBIC)),
                transforms.ToTensor()])
        )
    maps = {}
    for i in range(100):
        if i < 50:
            # training clients (0-49)
            maps.update({i: rotation_banks[i % 3]})
        else:
            # test clients (50-99)
            maps.update({i: rotation_banks[3 + (i % 3)]})
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
    shard_left = {i: shard_per_class for i in range(num_classes)}
    for i in range(num_users):
        available_class = [key for (key, value) in shard_left.items() if value > 0]
        try:
            selected_labels = np.random.choice(available_class, shard_per_client, replace=False)
            for l in selected_labels:
                shard_left[l] -= 1
        except:
            selected_labels = []
            while len(selected_labels) < shard_per_client:
                available_class = [key for (key, value) in shard_left.items() if value > 0]
                l = np.random.choice(available_class, 1, replace=True)[0]
                selected_labels.append(l)
                shard_left[l] -= 1
        rand_set = []
        for label in selected_labels:
            idx = np.random.choice(len(idx_dict[label]), replace=False)
            rand_set.append(idx_dict[label].pop(idx))
        dict_users[i] = np.concatenate(rand_set)
    for k, v in shard_left.items():
        assert v == 0
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
        if user_id < num_users - novel_clients:
            # Train and validation split ratio: 85:15
            random.shuffle(ids)
            train_ids = ids[:int(0.85 * len(ids))]
            train_dataset = CIFAR10_DATASET(X, Y, ids=train_ids, transform=maps[user_id])
            train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
            train_dataloaders[user_id] = train_dataloader

            validation_ids = ids[int(0.85 * len(ids)):]
            validation_dataset = CIFAR10_DATASET(X, Y, ids=validation_ids, transform=maps[user_id])
            validation_dataloader = DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
            validation_dataloaders[user_id] = validation_dataloader
            test_dataloaders[user_id] = None
        else:
            train_dataloaders[user_id] = None
            validation_dataloaders[user_id] = None
            test_dataset = CIFAR10_DATASET(X, Y, ids=ids, transform=maps[user_id])
            test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
            test_dataloaders[user_id] = test_dataloader
    users = [i for i in range(num_users)]
    return users, train_dataloaders, validation_dataloaders, test_dataloaders


def get_cifar10_dataloaders(num_users=100, batch_size=10, transform=None,
                            shard_per_client=2, novel_clients=10, ratio_of_test=1.0):
    # todo tdye: Note that the random seed might affect the model initialization, so reset it latter
    setup_seed(24)
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    y_train = y_train.reshape((50000,))
    y_test = y_test.reshape((10000,))
    x = np.concatenate((x_train, x_test), axis=0)
    y = np.concatenate((y_train, y_test), axis=0)

    table = PrettyTable(['TrainingX.', 'TrainingY.', 'TestX.', 'TestY.'])
    table.add_row([x_train.shape, y_train.shape, x_test.shape, y_test.shape])
    print(table)

    clients, train_loaders, validation_loaders, test_loaders = _get_cifar10_dataloaders(X=x, Y=y,
                                                                                        num_users=num_users,
                                                                                        batch_size=batch_size,
                                                                                        transform=transform,
                                                                                        novel_clients=novel_clients,
                                                                                        shard_per_client=shard_per_client,
                                                                                        ratio_of_test=ratio_of_test)
    return clients, train_loaders, validation_loaders, test_loaders


def get_rotated_cifar10_dataloaders(num_users=100, batch_size=10, shard_per_client=2, novel_clients=10):
    setup_seed(24)
    shard_per_client = 10
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    y_train = y_train.reshape((50000,))
    y_test = y_test.reshape((10000,))
    x = np.concatenate((x_train, x_test), axis=0)
    y = np.concatenate((y_train, y_test), axis=0)

    table = PrettyTable(['TrainingX.', 'TrainingY.', 'TestX.', 'TestY.'])
    table.add_row([x_train.shape, y_train.shape, x_test.shape, y_test.shape])
    print(table)

    clients, train_loaders, validation_loaders, test_loaders = _get_rotated_cifar10_dataloaders(X=x, Y=y,
                                                                                        num_users=num_users,
                                                                                        batch_size=batch_size,
                                                                                        novel_clients=novel_clients,
                                                                                        shard_per_client=shard_per_client)
    return clients, train_loaders, validation_loaders, test_loaders


def get_hetero_cifar10_dataloaders(num_users=100, batch_size=10, transform=None,
                            shard_per_client=2, novel_clients=10, client_train_ratio=0.85, client_val_ratio=0.15):
    setup_seed(24)
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    y_train = y_train.reshape((50000,))
    y_test = y_test.reshape((10000,))
    x = np.concatenate((x_train, x_test), axis=0)
    y = np.concatenate((y_train, y_test), axis=0)

    # divide 10% as the public dataset
    indices = np.array(range(y.shape[0]))
    np.random.shuffle(indices)
    public_size = int(y.shape[0] * 0.1)
    # print(public_size)
    public_x = x[indices[:public_size]]
    public_y = y[indices[:public_size]]
    x = x[indices[public_size:]]
    y = y[indices[public_size:]]
    #

    table = PrettyTable(['TrainingX.', 'TrainingY.', 'PublicX.', 'PublicY.'])
    table.add_row([x.shape, y.shape, public_x.shape, public_y.shape])
    print(table)

    clients, train_loaders, validation_loaders, test_loaders = _get_cifar10_dataloaders(X=x, Y=y,
                                                                                        num_users=num_users,
                                                                                        batch_size=batch_size,
                                                                                        transform=transform,
                                                                                        novel_clients=novel_clients,
                                                                                        shard_per_client=shard_per_client)
    public_dataset = SOFT_CIFAR10_DATASET(public_x, public_y, pseudo_Y=None, transform=transform)
    public_dataloader = DataLoader(dataset=public_dataset, batch_size=100, shuffle=False, num_workers=0)
    return clients, train_loaders, validation_loaders, test_loaders, public_dataloader
