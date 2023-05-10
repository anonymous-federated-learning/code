from torch.utils.data import Dataset, DataLoader
from utils.utils import *
from tensorflow import keras
from prettytable import PrettyTable


def load_data_from_keras():
    (x_train_fine, y_train_fine), (x_test_fine, y_test_fine) = keras.datasets.cifar100.load_data(label_mode='fine')
    (x_train_coarse, y_train_coarse), (x_test_coarse, y_test_coarse) = keras.datasets.cifar100.load_data(
        label_mode='coarse')

    return (x_train_fine, y_train_fine), (x_test_fine, y_test_fine), (x_train_coarse, y_train_coarse), (
        x_test_coarse, y_test_coarse)


class CIFAR100_DATASET(Dataset):
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


def get_cifar100_dataloaders(batch_size=10, transform=None, num_users=100, novel_clients=50):
    setup_seed(24)
    (x_train_fine, y_train_fine), (x_test_fine, y_test_fine), (x_train_coarse, y_train_coarse), (
        x_test_coarse, y_test_coarse) = load_data_from_keras()

    y_train_fine = y_train_fine.reshape((50000,))
    y_test_fine = y_test_fine.reshape((10000,))
    y_train_coarse = y_train_coarse.reshape((50000,))
    y_test_coarse = y_test_coarse.reshape((10000,))

    # print("x_train_fine", x_train_fine.shape, "y_train_fine", y_train_fine.shape)
    # print("x_test_fine", x_test_fine.shape, "y_test_fine", y_test_fine.shape)
    #
    # print("x_train_coarse", x_train_coarse.shape, "y_train_coarse", y_train_coarse.shape)
    # print("x_test_coarse", x_test_coarse.shape, "y_test_coarse", y_test_coarse.shape)

    table = PrettyTable(['TrainingX.', 'TrainingY.', 'TestX.', 'TestY.'])
    table.add_row([x_train_fine.shape, y_train_fine.shape, x_test_fine.shape, y_test_fine.shape])
    print(table)
    # x_coarse = np.concatenate((x_train_coarse, x_test_coarse), axis=0)
    y_coarse = np.concatenate((y_train_coarse, y_test_coarse), axis=0)

    x_fine = np.concatenate((x_train_fine, x_test_fine), axis=0)
    y_fine = np.concatenate((y_train_fine, y_test_fine), axis=0)
    # print(x_fine.shape, y.shape)

    # assert (x_train_fine == x_train_coarse).all()
    # assert (x_test_fine == x_test_coarse).all()

    coarse_group_ids = {i: [] for i in range(20)}
    for i in range(len(y_coarse)):
        coarse_l = y_coarse[i]
        coarse_group_ids[coarse_l].append(i)

    for key in coarse_group_ids.keys():
        random.shuffle(coarse_group_ids[key])

    dict_users = {i: None for i in range(num_users)}
    TOTAL_SAMPLES = 0
    for user in range(num_users):
        group_id = int(user / 5)
        drift = user % 5
        dataset_size = int(len(coarse_group_ids[group_id]) / 5)
        left = len(coarse_group_ids[group_id]) - 5 * dataset_size

        ids = coarse_group_ids[group_id][drift * dataset_size: (drift + 1) * dataset_size]
        if drift < left:
            ids.append(coarse_group_ids[-left + drift])
        TOTAL_SAMPLES += len(ids)
        dict_users.update({user: ids})
    assert TOTAL_SAMPLES == (50000 + 10000)
    train_dataloaders = {}
    validation_dataloaders = {}
    test_dataloaders = {}
    users = list(dict_users.keys())
    random.shuffle(users)
    for s_th, user_id in enumerate(users):  # s_th: id after shuffled
        ids = dict_users[user_id]
        if s_th < num_users - novel_clients:
            # Train and validation split ratio: 85:15
            random.shuffle(ids)
            train_ids = ids[:int(0.85 * len(ids))]
            train_dataset = CIFAR100_DATASET(x_fine, y_fine, ids=train_ids, transform=transform)
            train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
            train_dataloaders[user_id] = train_dataloader

            validation_ids = ids[int(0.85 * len(ids)):]
            validation_dataset = CIFAR100_DATASET(x_fine, y_fine, ids=validation_ids, transform=transform)
            validation_dataloader = DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=True,
                                               num_workers=0)
            validation_dataloaders[user_id] = validation_dataloader
            test_dataloaders[user_id] = None
        else:
            train_dataloaders[user_id] = None
            validation_dataloaders[user_id] = None
            test_dataset = CIFAR100_DATASET(x_fine, y_fine, ids=ids, transform=transform)
            test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
            test_dataloaders[user_id] = test_dataloader
    return users, train_dataloaders, validation_dataloaders, test_dataloaders


if __name__ == '__main__':
    users, train_dataloaders, validation_dataloaders, test_dataloaders = get_cifar100_dataloaders(batch_size=64)
    print(users)
    for u in users:
        for _, (x, y) in enumerate(train_dataloaders[u]):
            print(y)
        exit(0)
