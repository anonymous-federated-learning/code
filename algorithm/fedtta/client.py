from utils.setup_md import *
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import higher
from copy import deepcopy


class CLIENT:
    def __init__(self, user_id, train_loader, validation_loader, test_loader, config):
        self.config = config
        self.user_id = user_id
        self.device = torch.device(f"cuda:{config.cuda_no}") if config.cuda_no != -1 else torch.device("cpu")
        self.model = select_model(config=config)
        self.loss_nn = get_loss_nn(num_classes=config.num_classes)
        self.train_loader = train_loader
        self.iter_train_loader = None
        self.validation_loader = validation_loader
        self.iter_validation_loader = None
        self.test_loader = test_loader
        self.iter_test_loader = None
        self.loss_ce = nn.CrossEntropyLoss()
        self.stats = {
            'train-samples': 0,
            'test-samples': 0,
            'train-accuracy': 0,
            'test-accuracy': 0,
            'train-loss': None,
            'test-loss': None
        }

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

    def train(self, round_th):
        model, loss_nn = self.model, self.loss_nn
        model.to(self.device), loss_nn.to(self.device)
        model.train(), loss_nn.train()
        inner_lr = self.config.inner_lr * self.config.inner_lr_decay ** (round_th / self.config.inner_decay_step)
        outer_lr = self.config.outer_lr * self.config.outer_lr_decay ** (round_th / self.config.outer_decay_step)
        loss_nn_lr = self.config.loss_nn_lr * self.config.lossnn_lr_decay ** (round_th / self.config.lossnn_decay_step)
        inner_opt = optim.SGD(params=model.parameters(), lr=inner_lr, weight_decay=1e-4)
        # params = list(model.parameters()) + list(loss_nn.parameters())
        meta_opt = optim.SGD(
            [{'params': [param for name, param in model.named_parameters() if param.requires_grad],
              'lr': outer_lr},
             {'params': [param for name, param in loss_nn.named_parameters() if param.requires_grad],
              'lr': loss_nn_lr,
              'momentum': 0.9
              }],
            weight_decay=1e-4)

        mean_loss = []
        for it in range(self.config.local_iters):
            spt_x, spt_y = self.get_next_batch()
            spt_x, spt_y = spt_x.to(self.device), spt_y.to(self.device)
            with higher.innerloop_ctx(model, inner_opt, copy_initial_weights=False) as (fnet, diffopt):
                # Inner loop
                for _ in range(self.config.inner_iters):
                    spt_logits = fnet(spt_x)
                    spt_loss = loss_nn(spt_logits)
                    diffopt.step(spt_loss)
                eval_logits = fnet(spt_x)
                loss = self.loss_ce(eval_logits, spt_y)
                meta_opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
                torch.nn.utils.clip_grad_norm_(loss_nn.parameters(), 10)
                meta_opt.step()
                mean_loss.append(loss.item())
        model_params, loss_nn_params = self.get_params()

        if np.isnan(sum(mean_loss) / len(mean_loss)):
            print(f"client {self.user_id}, loss NAN")
            return 0, model_params, loss_nn_params, sum(mean_loss) / len(mean_loss)
            # exit(0)
        return self.train_samples_num, model_params, loss_nn_params, sum(mean_loss) / len(mean_loss)

    def finetune(self, model, loss_nn, round_th, data_loader):
        model, loss_nn = deepcopy(model), deepcopy(loss_nn)
        model.to(self.device), loss_nn.to(self.device)
        model.train(), loss_nn.eval()
        inner_lr = self.config.inner_lr * self.config.inner_lr_decay ** (round_th / self.config.inner_decay_step)
        inner_opt = optim.SGD(params=model.parameters(), lr=inner_lr, weight_decay=1e-4)
        for it in range(self.config.finetune_iters):
            spt_loss = torch.tensor(0.0).to(self.device)
            num_batches = 0
            for step, (spt_x, spt_y) in enumerate(data_loader):
                spt_x, spt_y = spt_x.to(self.device), spt_y.to(self.device)
                spt_logits = model(spt_x)
                spt_loss += loss_nn(spt_logits)
                num_batches += 1
            inner_opt.zero_grad()
            spt_loss /= num_batches
            spt_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            inner_opt.step()
        samples, acc, loss = self.test(model=model, data_loader=data_loader)
        return samples, acc, loss

    def fine_tuning_test(self, round_th, novel=False):
        if novel:
            test_samples, test_acc, test_loss = self.finetune(model=self.model, loss_nn=self.loss_nn, round_th=round_th, data_loader=self.test_loader)
            self.stats.update({
                'test-samples': test_samples,
                'test-accuracy': test_acc,
                'test-loss': test_loss,
                'gm-test-accuracy': 0, # gm_test_acc,
                'gm-test-loss': 0, # gm_test_loss
            })
        else:
            validation_samples, validation_acc, validation_loss = self.finetune(model=self.model, loss_nn=self.loss_nn, round_th=round_th, data_loader=self.validation_loader)

            self.stats.update({
                'train-samples': 1, # train_samples,
                'train-accuracy': 0, # train_acc,
                'train-loss': 0, # train_loss,
                'validation-samples': validation_samples,
                'validation-accuracy': validation_acc,
                'validation-loss': validation_loss,
                'gm-train-accuracy': 0, # gm_train_acc,
                'gm-train-loss': 0, # gm_train_loss,
                'gm-validation-accuracy': 0, # gm_validation_acc,
                'gm-validation-loss': 0, # gm_validation_loss
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

    def get_params(self):
        return deepcopy(self.model.cpu().state_dict()), deepcopy(self.loss_nn.cpu().state_dict())

    def set_params(self, model_params, loss_nn_params):
        self.model.load_state_dict(model_params)
        self.loss_nn.load_state_dict(loss_nn_params)

    def update(self, client):
        self.model.load_state_dict(client.model.state_dict())
        self.train_loader = client.train_loader
        self.test_loader = client.test_loader
