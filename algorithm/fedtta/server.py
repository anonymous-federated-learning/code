from copy import deepcopy

import torch

from utils.utils import *
from utils.setup_md import *
from algorithm.fedtta.client import CLIENT
from tqdm import tqdm
import time
import wandb


class SERVER:
    def __init__(self, config):
        self.config = config
        self.training_clients, self.novel_clients = self.setup_clients()
        self.selected_clients = []
        self.clients_model_params = []
        self.clients_loss_nn_params = []
        # affect server initialization
        setup_seed(config.seed)
        self.model = select_model(config=config)
        self.loss_nn = get_loss_nn(num_classes=config.num_classes)
        self.model_params = self.model.state_dict()
        self.loss_nn_params = self.loss_nn.state_dict()

        self.best_validation_acc = -1
        self.best_test_acc = -1

    def setup_clients(self):
        users, train_loaders, validation_loaders, test_loaders = setup_datasets(dataset=self.config.dataset,
                                                                                batch_size=self.config.batch_size,
                                                                                num_users=self.config.num_users,
                                                                                classes_per_client=self.config.classes_per_client,
                                                                                novel_clients=self.config.novel_users,
                                                                                training_clients=self.config.training_clients,
                                                                                train_alpha=self.config.train_alpha,
                                                                                novel_alpha=self.config.novel_alpha,
                                                                                ratio_of_test=self.config.ratio_of_test)
        clients = [
            CLIENT(user_id=user_id,
                   train_loader=train_loaders[user_id],
                   validation_loader=validation_loaders[user_id],
                   test_loader=test_loaders[user_id],
                   config=self.config)
            for user_id in users]
        # training_clients, novel_clients = clients[:-self.config.novel_users], clients[-self.config.novel_users:]
        training_clients, novel_clients = clients[:self.config.num_users-self.config.novel_users], clients[self.config.num_users-self.config.novel_users:]
        return training_clients, novel_clients

    def select_clients(self, round_th):
        np.random.seed(seed=self.config.seed + round_th)
        return np.random.choice(self.training_clients, self.config.clients_per_round, replace=False)

    def federate(self):
        print(f"Training with {len(self.training_clients)} clients!")
        for c in self.training_clients:
            print("Train", c.user_id, c.train_dataset_stats, end=' ')
            print("Validation", c.user_id, c.validation_dataset_stats)
        print(f"Test with {len(self.novel_clients)} novel clients!")
        for c in self.novel_clients:
            print("Test", c.user_id, c.test_dataset_stats)

        if self.config.load_pretrained_models:
            self.model = torch.load(f'./{self.config.path}/fedtta/{999}-model.pt')
            self.model_params = self.model.state_dict()
            self.loss_nn = torch.load(f'./{self.config.path}/fedtta/{999}-lossnn.pt')
            self.loss_nn_params = self.loss_nn.state_dict()
        for i in tqdm(range(self.config.num_rounds)):
            if self.config.load_pretrained_models:
                if i <= 999:
                    continue
            start_time = time.time()
            if not self.config.direct_test:
                self.selected_clients = self.select_clients(round_th=i)
                for k in range(len(self.selected_clients)):
                    c = self.selected_clients[k]
                    c.set_params(deepcopy(self.model_params), deepcopy(self.loss_nn_params))
                    train_samples_num, c_model_params, c_loss_nn_params, loss = c.train(round_th=i)
                    if train_samples_num != 0:
                        self.clients_model_params.append((train_samples_num, c_model_params))
                        self.clients_loss_nn_params.append((train_samples_num, c_loss_nn_params))
                self.model_params = fed_average(self.clients_model_params)
                self.loss_nn_params = fed_average(self.clients_loss_nn_params)
            end_time = time.time()
            print(f"training costs {end_time - start_time}(s)")
            if i >= self.config.start_evaluation and (i == 0 or (i + 1) % self.config.eval_interval == 0):
                if self.config.direct_test:
                    self.model = torch.load(f'./{self.config.path}/fedtta/{i}-model.pt')
                    self.model_params = self.model.state_dict()
                    self.loss_nn = torch.load(f'./{self.config.path}/fedtta/{i}-lossnn.pt')
                    self.loss_nn_params = self.loss_nn.state_dict()
                training_acc_list, training_loss_list, validation_acc_list, validation_loss_list, test_acc_list, test_loss_list, \
                gm_train_acc_list, gm_train_loss_list, gm_validation_acc_list, gm_validation_loss_list, gm_test_acc_list, gm_test_loss_list = self.test(
                    round_th=i)
                # print and log
                self.print_and_log(i, training_acc_list, training_loss_list,
                                   validation_acc_list, validation_loss_list,
                                   test_acc_list, test_loss_list,
                                   gm_train_acc_list, gm_train_loss_list,
                                   gm_validation_acc_list, gm_validation_loss_list,
                                   gm_test_acc_list, gm_test_loss_list)
                if self.config.save_models and not self.config.direct_test:
                    self.model.load_state_dict(self.model_params)
                    torch.save(self.model, f'./{self.config.path}/fedtta/{i}-model.pt')
                    self.loss_nn.load_state_dict(self.loss_nn_params)
                    torch.save(self.loss_nn, f'./{self.config.path}/fedtta/{i}-lossnn.pt')
            self.clients_model_params, self.clients_loss_nn_params = [], []

    def test(self, round_th):
        training_acc_list, training_loss_list = [], []
        validation_acc_list, validation_loss_list = [], []
        test_acc_list, test_loss_list = [], []
        gm_train_acc_list, gm_train_loss_list = [], []
        gm_validation_acc_list, gm_validation_loss_list = [], []
        gm_test_acc_list, gm_test_loss_list = [], []
        for c in self.training_clients:
            c.set_params(deepcopy(self.model_params), deepcopy(self.loss_nn_params))
            c.fine_tuning_test(round_th=round_th, novel=False)
            training_acc_list.append(
                (c.stats['train-samples'], c.stats['train-accuracy']))
            training_loss_list.append((c.stats['train-samples'], c.stats['train-loss']))
            validation_acc_list.append((c.stats['validation-samples'], c.stats['validation-accuracy']))
            validation_loss_list.append((c.stats['validation-samples'], c.stats['validation-loss']))
            gm_train_acc_list.append((c.stats['train-samples'], c.stats['gm-train-accuracy']))
            gm_train_loss_list.append((c.stats['train-samples'], c.stats['gm-train-loss']))
            gm_validation_acc_list.append((c.stats['validation-samples'], c.stats['gm-validation-accuracy']))
            gm_validation_loss_list.append((c.stats['validation-samples'], c.stats['gm-validation-loss']))

        for c in self.novel_clients:
            c.set_params(deepcopy(self.model_params), deepcopy(self.loss_nn_params))
            c.fine_tuning_test(round_th=round_th, novel=True)
            test_acc_list.append(
                (c.stats['test-samples'], c.stats['test-accuracy']))
            test_loss_list.append((c.stats['test-samples'], c.stats['test-loss']))
            gm_test_acc_list.append((c.stats['test-samples'], c.stats['gm-test-accuracy']))
            gm_test_loss_list.append((c.stats['test-samples'], c.stats['gm-test-loss']))
        return training_acc_list, training_loss_list, validation_acc_list, validation_loss_list, test_acc_list, test_loss_list, \
               gm_train_acc_list, gm_train_loss_list, gm_validation_acc_list, gm_validation_loss_list, gm_test_acc_list, gm_test_loss_list

    def print_and_log(self, round_th,
                      training_acc_list, training_loss_list,
                      validation_acc_list, validation_loss_list,
                      test_acc_list, test_loss_list,
                      gm_train_acc_list, gm_train_loss_list,
                      gm_validation_acc_list, gm_validation_loss_list,
                      gm_test_acc_list, gm_test_loss_list):
        training_acc = avg_metric(training_acc_list)
        training_loss = avg_metric(training_loss_list)
        validation_acc = avg_metric(validation_acc_list)
        validation_loss = avg_metric(validation_loss_list)
        test_acc = avg_metric(test_acc_list)
        test_loss = avg_metric(test_loss_list)

        gm_train_acc = avg_metric(gm_train_acc_list)
        gm_train_loss = avg_metric(gm_train_loss_list)
        gm_validation_acc = avg_metric(gm_validation_acc_list)
        gm_validation_loss = avg_metric(gm_validation_loss_list)
        gm_test_acc = avg_metric(gm_test_acc_list)
        gm_test_loss = avg_metric(gm_test_loss_list)

        # update best acc
        if validation_acc > self.best_validation_acc:
            self.best_validation_acc = validation_acc
        if test_acc > self.best_test_acc:
            self.best_test_acc = test_acc

        # post data error, encoder error, trainingAcc. format
        summary = {
            "round": round_th,
            "TrainAcc": training_acc,
            "ValidationAcc": validation_acc,
            "TestAcc": test_acc,
            "GMTrainAcc": gm_train_acc,
            "GMValidationAcc": gm_validation_acc,
            "GMTestAcc": gm_test_acc,
            "TrainLoss": training_loss,
            "ValidationLoss": validation_loss,
            "TestLoss": test_loss,
            "GMTrainLoss": gm_train_loss,
            "GMValidationLoss": gm_validation_loss,
            "GMTestLoss": gm_test_loss,
            "BestValidationAcc": self.best_validation_acc,
            "BestTestAcc": self.best_test_acc,

        }

        if self.config.use_wandb:
            wandb.log(summary)
        else:
            print(summary)
