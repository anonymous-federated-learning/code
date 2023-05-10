import torch
import numpy as np
import random
import json
import os
import hashlib
from collections import defaultdict
from copy import deepcopy

import argparse

ALGORITHMS = ['fedtta', 'fedtta_pp', 'fedavg', 'odpfl_hn', 'afl',
              'fedprox', 'fedadg', 'fedsr', 'fedgma', 'fedper_sampled', 'fedper_ensemble', 'scaffold',
              'hetero_fedtta', 'fedtta_prox', 'fedavg_tent', 'hetero_tent']
DATASETS = ['cifar10', 'cifar10_diri', 'mnist', 'cifar100', 'femnist', 'fashion_mnist', 'rotated_mnist']


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--use-wandb',
                        action='store_true',
                        default=False,
                        help='Use WandB?')

    parser.add_argument('--cuda-no',
                        help='cuda id, -1 for cpu.',
                        type=int,
                        default=0)

    parser.add_argument('--algorithm',
                        help='algorithm',
                        choices=ALGORITHMS,
                        required=True)

    parser.add_argument('--dataset',
                        help='name of dataset',
                        choices=DATASETS,
                        required=True)

    parser.add_argument('--model',
                        help='name of model',
                        type=str,
                        required=True)

    parser.add_argument('--num-rounds',
                        help='# of communication round',
                        type=int,
                        default=250)

    parser.add_argument('--eval-interval',
                        help='communication rounds between two evaluation',
                        type=int,
                        default=1)

    parser.add_argument('--clients-per-round',
                        help='# of selected clients per round',
                        type=int,
                        default=9)

    parser.add_argument('--local-iters',
                        help='# of iters',
                        type=int,
                        default=10)

    parser.add_argument('--inner-iters',
                        help='# of iters in the inner loop',
                        type=int,
                        default=1)

    parser.add_argument('--finetune-iters',
                        help='# of iters of the personalization in the finetune stage',
                        type=int,
                        default=1)

    parser.add_argument('--batch-size',
                        help='batch size when clients train on data',
                        type=int,
                        default=25)

    parser.add_argument('--loss-nn-lr',
                        help='learning rate of loss-nn',
                        type=float,
                        default=0.01)

    parser.add_argument('--lossnn-lr-decay',
                        help='decay rate for learning rate of loss_nn',
                        type=float,
                        default=0.99)

    parser.add_argument('--lossnn-decay-step',
                        help='decay step for learning rate of loss_nn',
                        type=int,
                        default=10)

    parser.add_argument('--seed',
                        help='seed for random client sampling and batch splitting',
                        type=int,
                        default=24)

    parser.add_argument('--num-users',
                        help='num users in total',
                        type=int,
                        default=100)

    parser.add_argument('--novel-users',
                        help='num of novel users',
                        type=int,
                        default=10)

    parser.add_argument('--training-clients',
                        help='num of training clients',
                        type=int,
                        default=50)

    parser.add_argument('--classes-per-client',
                        help='classes per client',
                        type=int,
                        default=2)

    # ODPFL-HN
    # --------------------------------------------------------------
    parser.add_argument('--encoder-lr',
                        help='learning rate of embedding encoder',
                        type=float,
                        default=0.01)

    parser.add_argument('--encoder-lr-decay',
                        help='decay rate for encoder learning rate',
                        type=float,
                        default=0.99)

    parser.add_argument('--encoder-decay-step',
                        help='decay step for encoder learning rate',
                        type=int,
                        default=10)

    # --------------------------------------------------------------
    parser.add_argument('--hnet-lr',
                        help='learning rate of hyper-network',
                        type=float,
                        default=0.01)

    parser.add_argument('--hnet-lr-decay',
                        help='decay rate for hyper-net learning rate',
                        type=float,
                        default=0.99)

    parser.add_argument('--hnet-decay-step',
                        help='decay step for hyper-net learning rate',
                        type=int,
                        default=10)

    # --------------------------------------------------------------
    parser.add_argument('--local-lr',
                        help='learning rate of client local network',
                        type=float,
                        default=0.01)

    parser.add_argument('--local-lr-decay',
                        help='decay rate for local-net learning rate',
                        type=float,
                        default=0.99)

    parser.add_argument('--local-decay-step',
                        help='decay step for local-net learning rate',
                        type=int,
                        default=10)

    # FedTTA
    parser.add_argument('--inner-lr',
                        help='learning rate for inner update',
                        type=float,
                        default=0.01)

    parser.add_argument('--inner-lr-decay',
                        help='decay rate for inner learning rate',
                        type=float,
                        default=0.99)

    parser.add_argument('--inner-decay-step',
                        help='decay step for inner learning rate',
                        type=int,
                        default=10)

    parser.add_argument('--outer-lr',
                        help='learning rate for outer update',
                        type=float,
                        default=0.01)

    parser.add_argument('--outer-lr-decay',
                        help='decay rate for outer learning rate',
                        type=float,
                        default=0.99)

    parser.add_argument('--outer-decay-step',
                        help='decay step for outer learning rate',
                        type=int,
                        default=10)
    # AFL
    parser.add_argument('--lr-omega',
                        help='learning rate of model parameters omega of AFL',
                        type=float,
                        default=0.01)

    parser.add_argument('--lr-omega-decay',
                        help='decay rate for lr-omega',
                        type=float,
                        default=0.99)

    parser.add_argument('--omega-decay-step',
                        help='decay step for lr-omega',
                        type=int,
                        default=10)

    parser.add_argument('--lr-lambda',
                        help='learning rate of mixture coefficient of AFL',
                        type=float,
                        default=0.01)

    parser.add_argument('--lr-lambda-decay',
                        help='decay rate for lr-lambda',
                        type=float,
                        default=0.99)

    parser.add_argument('--lambda-decay-step',
                        help='decay step for lr-lambda',
                        type=int,
                        default=10)

    # FedTTA+Stopping
    parser.add_argument('--patience',
                        help='patience',
                        type=int,
                        default=5)

    # FedProx
    parser.add_argument('--mu',
                        help='coefficient of proximal term in FedProx',
                        type=float,
                        default=0.1)

    # FedAvg & FedProx
    parser.add_argument('--lr',
                        help='learning rate of FedAvg ...',
                        type=float,
                        default=0.01)

    parser.add_argument('--lr-decay',
                        help='decay rate for lr',
                        type=float,
                        default=0.99)

    parser.add_argument('--decay-step',
                        help='decay step for lr',
                        type=int,
                        default=10)

    # FedADG
    parser.add_argument('--lr-generator',
                        help='learning rate for generator in FedADG',
                        type=float,
                        default=0.1)

    parser.add_argument('--lr-generator-decay',
                        help='decay rate for generator in FedADG',
                        type=float,
                        default=0.99)

    parser.add_argument('--generator-decay-step',
                        help='decay step for of generator in FedADG',
                        type=int,
                        default=10)

    parser.add_argument('--lr-discriminator',
                        help='learning rate for discriminator in FedADG',
                        type=float,
                        default=0.1)

    parser.add_argument('--lr-discriminator-decay',
                        help='decay rate for lr discriminator in FedADG',
                        type=float,
                        default=0.99)

    parser.add_argument('--discriminator-decay-step',
                        help='decay step for lr of discriminator in FedADG',
                        type=int,
                        default=10)

    parser.add_argument('--D-beta',
                        help='coefficient of training net in FedADG',
                        type=float,
                        default=1)

    # FedSR
    parser.add_argument('--L2R-coeff',
                        help='coefficient of L2R in FedSR',
                        type=float,
                        default=1e-2)

    parser.add_argument('--CMI-coeff',
                        help='coefficient of CMI in FedSR',
                        type=float,
                        default=5e-4)

    # FedGMA
    parser.add_argument('--server-lr',
                        help='server learning rate in FedGMA',
                        type=float,
                        default=0.1)

    parser.add_argument('--server-lr-decay',
                        help='decay rate for server lr in FedGMA',
                        type=float,
                        default=0.99)

    parser.add_argument('--serverlr-decay-step',
                        help='decay step for serverlr in FedGMA',
                        type=int,
                        default=10)

    parser.add_argument('--threshold',
                        help='threshold for masking in FedGMA',
                        type=float,
                        default=0.0)

    # FedPer_Sampled
    parser.add_argument('--share-layers',
                        help='share layers in FedPer',
                        type=int,
                        default=2)
    # FedPer_Ensemble
    parser.add_argument('--ratio-random-clients',
                        help='ratio of random clients for ensemble in FedPer_Ensemble',
                        type=float,
                        default=0.5)

    # SCAFFOLD
    parser.add_argument('--global-lr',
                        help='global learning rate of SCAFFOLD',
                        type=float,
                        default=1.0
                        )

    # FedTTA_Noise
    parser.add_argument('--tta-prox',
                        help='coefficient of the prox loss minimizing the distribution distance between the spt_logits and the prox_logits',
                        type=float,
                        default=0.01)

    # Dirichlet
    parser.add_argument('--train-alpha',
                        help='alpha of dirichlet for partitioning training datasets',
                        type=float,
                        default=0.5)

    parser.add_argument('--novel-alpha',
                        help='alpha of dirichlet for partitioning test datasets',
                        type=float,
                        default=0.1)

    parser.add_argument('--start-evaluation',
                        help='when to start evaluation',
                        type=int,
                        default=0)

    # Dirichlet test
    parser.add_argument('--save-models',
                        action='store_true',
                        default=False,
                        help='Save models?')

    parser.add_argument('--path',
                        type=str,
                        default='diri_models',
                        help='note for save models path')

    parser.add_argument('--direct-test',
                        action='store_true',
                        default=False,
                        help='Directly test with pretrained model stored')

    parser.add_argument('--load-pretrained-models',
                        action='store_true',
                        default=False,
                        help='Load pretrained models')

    parser.add_argument('--local-distillation-epochs',
                        help='local distillation steps',
                        type=int,
                        default=1)

    parser.add_argument('--ratio-of-test',
                        help='small size test clients',
                        type=float,
                        default=1.0)

    return parser.parse_args()

class HparamsGen(object):
    def __init__(self, name, default, gen_fn=None):
        self.name = name
        self.default = default
        self.gen_fn = gen_fn
    def __call__(self,hparams_gen_seed=0):
        if hparams_gen_seed == 0 or self.gen_fn is None:
            return self.default
        else:
            s = f"{hparams_gen_seed}_{self.name}"
            seed = int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16) % (2**31)
            return self.gen_fn(np.random.RandomState(seed))

def setup_seed(rs):
    """
    set random seed for reproducing experiments
    :param rs: random seed
    :return: None
    """
    torch.manual_seed(rs)
    torch.cuda.manual_seed_all(rs)
    np.random.seed(rs)
    random.seed(rs)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def read_dir(data_dir):
    clients = []
    data = defaultdict(lambda: None)

    files = os.listdir(data_dir)
    files = [f for f in files if f.endswith('.json')]
    for f in files:
        file_path = os.path.join(data_dir, f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        clients.extend(cdata['users'])
        data.update(cdata['user_data'])

    # clients = list(sorted(data.keys()))
    return clients, data


def read_data(train_data_dir, test_data_dir):
    """parses data in given train and test data directories
    assumes:
    - the data in the input directories are .json files with
        keys 'users' and 'user_data'
    - the set of train set users is the same as the set of test set users
    Return:
        clients: list of client ids
        groups: list of group ids; empty list if none found
        train_data: dictionary of train data
        test_data: dictionary of test data
    """
    train_clients, train_data = read_dir(train_data_dir)
    test_clients, test_data = read_dir(test_data_dir)
    assert train_clients.sort() == test_clients.sort()

    return train_clients, train_data, test_data


def fed_average_hnet(updates):
    total_weight = 0
    for item in updates:
        (client_samples_num, model_update) = item[0], item[1]
        total_weight += client_samples_num
    len_tuple = len(updates[0][1])
    average_update = [0] * len_tuple
    for t_id in range(len_tuple):
        for i in range(0, len(updates)):
            client_samples, model_update = updates[i][0], updates[i][1]
            # weight
            w = client_samples / total_weight
            if i == 0:
                average_update[t_id] = model_update[t_id] * w
            else:
                average_update[t_id] += model_update[t_id] * w
    # return global model params
    return tuple(average_update)


def fed_average(updates):
    total_weight = 0
    (client_samples_num, new_params) = updates[0][0], updates[0][1]

    for item in updates:
        (client_samples_num, client_params) = item[0], item[1]
        total_weight += client_samples_num

    for k in new_params.keys():
        for i in range(0, len(updates)):
            client_samples, client_params = updates[i][0], updates[i][1]
            # weight
            w = client_samples / total_weight
            if i == 0:
                new_params[k] = client_params[k] * w
            else:
                new_params[k] += client_params[k] * w
    # return global model params
    return new_params


def avg_metric(metric_list):
    total_weight = 0
    total_metric = 0
    for (samples_num, metric) in metric_list:
        total_weight += samples_num
        total_metric += samples_num * metric
    average = total_metric / total_weight

    return average


def get_pseudo_g(old_state_dict, new_state_dict):
    # old - new
    pseudo_g = deepcopy(new_state_dict)
    for k in new_state_dict.keys():
        pseudo_g[k] = old_state_dict[k] - new_state_dict[k]
    # return pseudo_g
    return deepcopy(pseudo_g)


def project(y):
    ''' algorithm comes from:
    https://arxiv.org/pdf/1309.1541.pdf
    '''
    u = sorted(y, reverse=True)
    x = []
    rho = 0
    for i in range(len(y)):
        if (u[i] + (1.0/(i+1)) * (1-np.sum(np.asarray(u)[:i+1]))) > 0:
            rho = i + 1
    lambda_ = (1.0/rho) * (1-np.sum(np.asarray(u)[:rho]))
    for i in range(len(y)):
        x.append(max(y[i]+lambda_, 0))
    return x

num_classes = {'cifar10': 10,
               'cifar10_diri': 10,
                   'mnist': 10,
               'rotated_mnist': 10,
                   'cifar100': 100,
                   'femnist': 62,
               'fashion_mnist': 10}

def tensorboard_smoothing(x,smooth=0.99):
    x = x.copy()
    weight = smooth
    for i in range(1,len(x)):
        x[i] = (x[i-1] * weight + x[i]) / (weight + 1)
        weight = (weight + 1) * smooth
    return x

