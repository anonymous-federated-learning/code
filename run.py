from utils.utils import *
from algorithm.fedtta.server import SERVER as FedTTA_SERVER
from algorithm.fedtta_prox.server import SERVER as FedTTA_Prox_SERVER
from algorithm.fedtta_pp.server import SERVER as FedTTA_PP_SERVER
from algorithm.hetero_tent.server import SERVER as Hetero_TENT_SERVER
from algorithm.hetero_fedtta.server import SERVER as Hetero_FedTTA_SERVER

if __name__ == '__main__':
    args = parse_args()
    if args.use_wandb:
        import wandb

        wandb.init(project="xxx", entity="xxx")
        wandb.watch_called = False
        config = wandb.config
        config.update(args)
    else:
        config = args

    num_classes = {'cifar10': 10,
                   'cifar10_diri': 10,
                   'mnist': 10,
                   'cifar100': 100,
                   'femnist': 62,
                   'fashion_mnist': 10,
                   'rotated_mnist': 10,
                   'rotated_cifar10': 10}
    config.num_classes = num_classes[config.dataset]

    server = None
    if config.algorithm == 'fedtta':
        server = FedTTA_SERVER(config=config)
    elif config.algorithm == 'fedtta_prox':
        server = FedTTA_Prox_SERVER(config=config)
    elif config.algorithm == 'fedtta_pp':
        server = FedTTA_PP_SERVER(config=config)
    elif config.algorithm == 'hetero_tent':
        server = Hetero_TENT_SERVER(config=config)
    elif config.algorithm == 'hetero_fedtta':
        server = Hetero_FedTTA_SERVER(config=config)
    server.federate()
