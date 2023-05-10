from utils.setup_md import *
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import higher
import torch.nn.functional as F
from copy import deepcopy


class CLIENT:
    def __init__(self, user_id, train_loader, validation_loader, test_loader, config):
        self.config = config
        self.user_id = user_id
        self.device = torch.device(f"cuda:{config.cuda_no}") if config.cuda_no != -1 else torch.device("cpu")
        self.train_loader = train_loader
        self.iter_train_loader = None
        self.validation_loader = validation_loader
        self.iter_validation_loader = None
        self.test_loader = test_loader
        self.iter_test_loader = None
        model_type = ['small', 'medium', 'big'][self.user_id % 3]
        self.model = select_hetero_model(config=config, model_type=model_type)
        self.loss_ce = nn.CrossEntropyLoss()
        self.loss_kl = nn.KLDivLoss(reduction="batchmean")
        self.stats = {
            'train-samples': 0,
            'test-samples': 0,
            'train-accuracy': 0,
            'test-accuracy': 0,
            'train-loss': None,
            'test-loss': None
        }
        self.public_dataloader = None

    def reset_models(self):
        model_type = ['small', 'medium', 'big'][self.user_id % 3]
        self.model = select_hetero_model(config=self.config, model_type=model_type)

    @property
    def train_samples_num(self):
        return len(self.train_loader.dataset) if self.train_loader else None

    @property
    def validation_samples_num(self):
        return len(self.validation_loader.dataset) if self.validation_loader else None

    @property
    def test_samples_num(self):
        return len(self.test_loader.dataset) if self.test_loader else None

    @property
    def train_dataset_stats(self):
        if self.train_loader:
            try:
                Y = self.train_loader.dataset.Y[self.train_loader.dataset.ids]
            except:
                Y = self.train_loader.dataset.Y
            unique_labels = np.unique(Y)
            res = {}
            for l in unique_labels:
                count = len(np.where(Y == l)[0])
                res.update({l: count})
        else:
            res = "Surrogate!"
        return res

    @property
    def validation_dataset_stats(self):
        if self.validation_loader:
            try:
                Y = self.validation_loader.dataset.Y[self.validation_loader.dataset.ids]
            except:
                Y = self.validation_loader.dataset.Y
            unique_labels = np.unique(Y)
            res = {}
            for l in unique_labels:
                count = len(np.where(Y == l)[0])
                res.update({l: count})
        else:
            res = "Surrogate!"
        return res

    @property
    def test_dataset_stats(self):
        if self.test_loader:
            try:
                Y = self.test_loader.dataset.Y[self.test_loader.dataset.ids]
            except:
                Y = self.test_loader.dataset.Y
            unique_labels = np.unique(Y)
            res = {}
            for l in unique_labels:
                count = len(np.where(Y == l)[0])
                res.update({l: count})
        else:
            res = "Surrogate!"
        return res

    def get_next_batch(self):
        if not self.iter_train_loader:
            self.iter_train_loader = iter(self.train_loader)
        try:
            (X, y) = next(self.iter_train_loader)
        except StopIteration:
            self.iter_train_loader = iter(self.train_loader)
            (X, y) = next(self.iter_train_loader)
        return X, y

    def get_next_test_batch(self):
        if not self.iter_test_loader:
            self.iter_test_loader = iter(self.test_loader)
        try:
            (X, y) = next(self.iter_test_loader)
        except StopIteration:
            self.iter_test_loader = iter(self.test_loader)
            (X, y) = next(self.iter_test_loader)
        return X, y

    def train(self, round_th, digest=False, revisit=False):
        distillation_epochs = self.config.local_distillation_epochs
        if revisit is False:
            distillation_epochs = distillation_epochs * 2
        model = self.model
        model.to(self.device)
        model.train()
        lr = self.config.lr * self.config.lr_decay ** (round_th / self.config.decay_step)
        optimizer = optim.SGD(
            [{'params': [param for name, param in model.named_parameters() if param.requires_grad],
              'lr': lr}],
            # momentum=0.9,
            weight_decay=1e-4)
        if digest: #
            # Aligns with public average logits, digest
            for it in range(distillation_epochs):
                for _, all_ in enumerate(self.public_dataloader):
                    x, y, personalized_avg_logits = all_[0], all_[1], all_[2]
                    x, personalized_avg_logits = x.to(self.device), personalized_avg_logits.to(self.device)
                    spt_logits = model(x)
                    distillation_loss = self.loss_kl(F.log_softmax(spt_logits, dim=1), F.softmax(personalized_avg_logits.detach(), dim=1))
                    optimizer.zero_grad()
                    distillation_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
                    optimizer.step()
        if revisit:
            mean_loss = [] # revisit
            for it in range(self.config.local_iters):
                spt_x, spt_y = self.get_next_batch()
                spt_x, spt_y = spt_x.to(self.device), spt_y.to(self.device)
                spt_logits = model(spt_x)
                spt_loss = self.loss_ce(spt_logits, spt_y)
                optimizer.zero_grad()
                spt_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
                optimizer.step()
                mean_loss.append(spt_loss.item())
            model = self.get_nns()

            if np.isnan(sum(mean_loss) / len(mean_loss)):
                print(f"client {self.user_id}, loss NAN")
                return 0, sum(mean_loss) / len(mean_loss)
                # exit(0)
            return self.train_samples_num, model, sum(mean_loss) / len(mean_loss)

    def finetune(self, model, round_th, data_loader, return_finetuned_model=False):
        model = deepcopy(model)
        model.to(self.device)
        model.train()
        lr = self.config.lr * self.config.lr_decay ** (round_th / self.config.decay_step)
        opt = optim.SGD(params=model.parameters(), lr=lr, weight_decay=1e-4)
        probabilities = []
        for _, all_ in enumerate(data_loader):
            spt_x, y = all_[0], all_[1]
            spt_x = spt_x.to(self.device)
            aug_p = F.softmax(model(spt_x), dim=1)
            probabilities.append(aug_p)
        tmp = torch.cat(probabilities)
        b = tmp * torch.log2(tmp)
        tent_loss = -1.0 * b.sum() / b.shape[0]  # memo loss: entropy
        opt.zero_grad()
        tent_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        opt.step()
        if return_finetuned_model:
            return deepcopy(model.cpu())
        samples, acc, loss = self.test(model=model, data_loader=data_loader)
        return samples, acc, loss

    def fine_tuning_test(self, round_th, novel=False):
        if novel:
            test_samples, test_acc, test_loss = self.finetune(model=self.model,
                                                              round_th=round_th,
                                                              data_loader=self.test_loader)
            self.stats.update({
                'test-samples': test_samples,
                'test-accuracy': test_acc,
                'test-loss': test_loss,
                'gm-test-accuracy': 0,  # gm_test_acc,
                'gm-test-loss': 0,  # gm_test_loss
            })
        else:
            validation_samples, validation_acc, validation_loss = self.finetune(model=self.model,
                                                                                round_th=round_th,
                                                                                data_loader=self.validation_loader)

            self.stats.update({
                'train-samples': 1,  # train_samples,
                'train-accuracy': 0,  # train_acc,
                'train-loss': 0,  # train_loss,
                'validation-samples': validation_samples,
                'validation-accuracy': validation_acc,
                'validation-loss': validation_loss,
                'gm-train-accuracy': 0,  # gm_train_acc,
                'gm-train-loss': 0,  # gm_train_loss,
                'gm-validation-accuracy': 0,  # gm_validation_acc,
                'gm-validation-loss': 0,  # gm_validation_loss
            })

    def test(self, model=None, data_loader=None):
        model.eval()
        model.to(self.device)

        total_right = 0
        total_samples = 0
        mean_loss = []
        with torch.no_grad():
            for step, (data, labels) in enumerate(data_loader):
                data = data.to(self.device)
                labels = labels.to(self.device)
                output = model(data)
                loss = self.loss_ce(output, labels)
                mean_loss.append(loss.item())
                output = torch.argmax(output, dim=-1)
                total_right += torch.sum(output == labels)
                total_samples += len(labels)
            acc = float(total_right) / total_samples

        return total_samples, acc, sum(mean_loss) / len(mean_loss)

    def get_nns(self):
        return deepcopy(self.model)

    def set_global_info(self, public_dataloader):
        self.public_dataloader = public_dataloader

    def update(self, client):
        self.model.load_state_dict(client.model.state_dict())
        self.train_loader = client.train_loader
        self.test_loader = client.test_loader
