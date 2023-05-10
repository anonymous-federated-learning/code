from copy import deepcopy
from utils.utils import *
from utils.setup_md import *
from algorithm.hetero_fedtta.client import CLIENT
from tqdm import tqdm
import higher
import torch
import torch.nn as nn
import time
import wandb
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class SOFT_DATASET(Dataset):
    def __init__(self, X, Y, base_Y=None, personalzied_Y=None, transform=None):
        self.X = X
        self.Y = Y
        self.base_Y = base_Y
        self.personalized_Y = personalzied_Y
        self.transform = transform

    def __getitem__(self, item):
        x, y = self.X[item], self.Y[item]
        base_y = self.base_Y[item] if self.base_Y is not None else -1
        personalized_y = self.personalized_Y[item] if self.personalized_Y is not None else -1
        if self.transform is not None:
            x = self.transform(x)
        else:
            x = torch.tensor(x).float()
        return x, y, base_y, personalized_y

    def __len__(self):
        return self.Y.shape[0]

class SERVER:
    def __init__(self, config):
        self.config = config
        self.public_dataloader = None
        self.training_clients, self.novel_clients = self.setup_clients()
        self.device = torch.device(f"cuda:{config.cuda_no}") if config.cuda_no != -1 else torch.device("cpu")

        self.selected_clients = []


        # affect server initialization
        setup_seed(config.seed)
        # self.model = select_hetero_model(config=config, model_type='big')
        # self.loss_nn = get_hetero_loss_nn(num_classes=config.num_classes, model_type='big')
        self.loss_kl = nn.KLDivLoss(reduction="batchmean")

        self.best_validation_acc = -1
        self.best_test_acc = -1

    def setup_clients(self):
        users, train_loaders, validation_loaders, test_loaders, public_dataloader = setup_hetero_datasets(dataset=self.config.dataset,
                                                                                batch_size=self.config.batch_size,
                                                                                num_users=self.config.num_users,
                                                                                classes_per_client=self.config.classes_per_client,
                                                                                novel_clients=self.config.novel_users)
        self.public_dataloader = public_dataloader
        clients = [
            CLIENT(user_id=user_id,
                   train_loader=train_loaders[user_id],
                   validation_loader=validation_loaders[user_id],
                   test_loader=test_loaders[user_id],
                   config=self.config)
            for user_id in users]
        training_clients, novel_clients = clients[:-self.config.novel_users], clients[-self.config.novel_users:]
        return training_clients, novel_clients

    def select_clients(self, round_th):
        np.random.seed(seed=self.config.seed + round_th)
        return np.random.choice(self.training_clients, self.config.clients_per_round, replace=False)

    def ensemble_knowledge(self, clients, round_th):
        public_dataloader = deepcopy(self.public_dataloader)
        # 先finetune
        original_models = []
        finetuned_models = []
        for c in clients:
            original_models.append(deepcopy(c.model))
            return_model = c.finetune(model=c.model, loss_nn=c.loss_nn, round_th=round_th, data_loader=public_dataloader, return_finetuned_model=True)
            finetuned_models.append(return_model)


        print("\naggregating base model knowledge")
        base_ensemble_logits = []
        # base_clients_acc = []
        for m in tqdm(original_models):
            m.to(self.device)
            m.eval()
            m_logits = []
            with torch.no_grad():
                for _, all_ in enumerate(public_dataloader):
                    x, y = all_[0].to(self.device), all_[1].to(self.device)
                    tmp_logits = m(x)
                    m_logits.append(tmp_logits.detach())
            m_all_logits = torch.cat(m_logits)

            def cal_acc(logits):
                output = torch.argmax(logits, dim=-1).cpu()
                targets = torch.tensor(public_dataloader.dataset.Y)
                acc = torch.sum(output == targets) / targets.shape[0]
                return acc.item()

            # base_m_acc = cal_acc(m_all_logits)
            # base_clients_acc.append(base_m_acc)
            base_ensemble_logits.append(m_all_logits)
        base_ensemble_logits = torch.cat(base_ensemble_logits, dim=0).reshape((-1, *base_ensemble_logits[0].shape))
        base_avg_logits = torch.mean(base_ensemble_logits, dim=0)

        print("\naggregating personalized model knowledge")
        ensemble_logits = []
        # clients_acc = []
        for m in tqdm(finetuned_models):
            m.to(self.device)
            m.eval()
            m_logits = []
            with torch.no_grad():
                for _, all_ in enumerate(public_dataloader):
                    x, y = all_[0].to(self.device), all_[1].to(self.device)
                    tmp_logits = m(x)
                    m_logits.append(tmp_logits.detach())
            m_all_logits = torch.cat(m_logits)
            def cal_acc(logits):
                output = torch.argmax(logits, dim=-1).cpu()
                targets = torch.tensor(public_dataloader.dataset.Y)
                acc = torch.sum(output == targets) / targets.shape[0]
                return acc.item()
            # m_acc = cal_acc(m_all_logits)
            # clients_acc.append(m_acc)
            ensemble_logits.append(m_all_logits)
        ensemble_logits = torch.cat(ensemble_logits, dim=0).reshape((-1, *ensemble_logits[0].shape))
        personalized_avg_logits = torch.mean(ensemble_logits, dim=0)
        public_x, public_y, public_base_avg_logits, public_personalized_avg_logits, transform = public_dataloader.dataset.X, public_dataloader.dataset.Y, base_avg_logits, personalized_avg_logits, public_dataloader.dataset.transform
        new_public_dataset = SOFT_DATASET(X=public_x, Y=public_y, base_Y=public_base_avg_logits, personalzied_Y=public_personalized_avg_logits, transform=transform)
        base_ensemble_acc = cal_acc(public_base_avg_logits)
        personalized_ensemble_acc = cal_acc(public_personalized_avg_logits)
        self.public_dataloader = DataLoader(dataset=new_public_dataset, batch_size=100, shuffle=False, num_workers=0)
        return base_ensemble_acc, personalized_ensemble_acc
    def federate(self):
        print(f"Training with {len(self.training_clients)} clients!")
        for c in self.training_clients:
            print("Train", c.user_id, c.train_dataset_stats, end=' ')
            print("Validation", c.user_id, c.validation_dataset_stats)
        print(f"Test with {len(self.novel_clients)} novel clients!")
        for c in self.novel_clients:
            print("Test", c.user_id, c.test_dataset_stats)

        for i in tqdm(range(self.config.num_rounds)):
            start_time = time.time()
            self.selected_clients = self.select_clients(round_th=i)
            # todo(tdye): distill to a big model for transferring ...
            for k in range(len(self.selected_clients)):
                c = self.selected_clients[k]
                c.set_global_info(public_dataloader=deepcopy(self.public_dataloader))
                if i == 0:
                    train_samples_num, c_model_params, c_loss_nn_params, loss = c.train(round_th=i, digest=False, revisit=True)
                else:
                    train_samples_num, c_model_params, c_loss_nn_params, loss = c.train(round_th=i, digest=True, revisit=True)
            base_ensemble_public_acc, personalized_ensemble_public_acc = self.ensemble_knowledge(clients=self.selected_clients, round_th=i)
            wandb.log({
                "round": i,
                "BaseEnsemblePublicAcc": base_ensemble_public_acc,
                "EnsemblePublicAcc": personalized_ensemble_public_acc
            })

            end_time = time.time()
            print(f"training costs {end_time - start_time}(s)")
            if i == 0 or (i + 1) % self.config.eval_interval == 0:
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


    def test(self, round_th):
        training_acc_list, training_loss_list = [], []
        validation_acc_list, validation_loss_list = [], []
        test_acc_list, test_loss_list = [], []
        gm_train_acc_list, gm_train_loss_list = [], []
        gm_validation_acc_list, gm_validation_loss_list = [], []
        gm_test_acc_list, gm_test_loss_list = [], []
        for c in self.training_clients:
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
            c.public_dataloader = deepcopy(self.public_dataloader)
            c.reset_models()
            c.train(round_th=round_th, digest=True, revisit=False)
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
