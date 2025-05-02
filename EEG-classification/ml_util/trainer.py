import torch
from torch import nn
from typing import Callable, List, Tuple, Dict, Any
from torch.utils.data import DataLoader
from ml_util.data_module import calculate_accuracy

class GoodClassificationModel():
    def __init__(self, 
                 model: nn.Module, 
                 loss: Callable, 
                 hyperparameters: Dict[str, Any],
                 device: torch.device,
                 nblabels: int
                 ):
        '''
        Module should end with an output layer from which loss can be applied
        '''
        self.model = model.to(device)
        self.loss = loss
        self.hyperparameters = hyperparameters
        self.device = device
        self.nblabels = nblabels
        self.epoch_log = {}
        self.epoch = 0

    def epoch_begin_callback(self):
        self.epoch_log = {}

    def training_step(self, batch):
        # Returns what needs to be minimized
        self.model.train()
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        scores = self.model(x)
        loss = self.loss(scores, y)

        self.log("train_preds", torch.argmax(scores, dim=1))
        self.log("train_labels", y)
        self.log("train_loss", (loss, len(batch)))

        return loss

    def validation_step(self, batch, train=True):
        keyword = "val" if train else "test"
        self.model.eval()
        with torch.no_grad():
            x, y = batch
            x, y = x.to(self.device), y.to(self.device)
            scores = self.model(x)
            loss = self.loss(scores, y)

            self.log(f"{keyword}_preds", torch.argmax(scores, dim=1))
            self.log(f"{keyword}_labels", y)
            self.log(f"{keyword}_loss", (loss, len(batch)))
            self.log("num_classes", scores.shape[1])

    def epoch_end_callback(self, train):
        if train:
            self.epoch += 1

    def log(self, key, value):
        '''
        This is an epoch log
        '''
        if key in self.epoch_log:
            self.epoch_log[key].append(value)
        else:
            self.epoch_log[key] = [value]

    def epoch_results(self):
        res = {}
        res["epoch"] = self.epoch
        res["num_classes"] = self.epoch_log["num_classes"][0]

        for part in ["train", "val", "test"]:
            if f"{part}_loss" in self.epoch_log: res[f"{part}_loss"] = self.calculate_loss(f"{part}_loss") 
            if f"{part}_preds" in self.epoch_log and f"{part}_labels" in self.epoch_log: 
                concat_preds, concat_labels = self.concat_all_preds_labels(f"{part}_preds", f"{part}_labels")
                res[f"{part}_acc"] = calculate_accuracy(concat_preds, concat_labels)
            if f"{part}_preds" in self.epoch_log and f"{part}_labels" in self.epoch_log:
                res[f"{part}_confusion"] = self.concat_all_preds_labels(f"{part}_preds", f"{part}_labels")
        return res
    
    def concat_all_preds_labels(self, keypreds, keylabels):
        if keypreds in self.epoch_log and keylabels in self.epoch_log:
            preds = torch.cat(self.epoch_log[keypreds])
            labels = torch.cat(self.epoch_log[keylabels])
            return preds, labels
        else:
            raise RuntimeError(f"{keypreds} or {keylabels} not logged during epoch calculation")

    def calculate_loss(self, key):
        if key in self.epoch_log:
            res = 0
            total = 0
            for a, b in self.epoch_log[key]:
                res += a * b
                total += b
            return res / total
        else:
            raise RuntimeError(f"{key} not logged during epoch calculation")
    
    def configure_optimizer(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.hyperparameters["lr"], weight_decay=self.hyperparameters['weight_decay'])

class GoodTrainer:
    def __init__(self):
        self.optimizer = None
        self.loggers = []
        self.babysitters = []

    def train(self, goodModel, dataset, nb_epochs):
        if not(hasattr(goodModel, "optimizer")):
            goodModel.optimizer = goodModel.configure_optimizer()

        for epoch in range(nb_epochs):
            goodModel.epoch_begin_callback()    # Initializes the epoch variables
            goodModel.epoch = epoch
            for babysitter in self.babysitters: # Applies babysitting procedures
                babysitter(goodModel)
            change_optimizer_parameters(goodModel.optimizer, goodModel.hyperparameters)

            batch_size = goodModel.hyperparameters["batch_size"]
            train_dataloader = DataLoader(dataset.train_dataset, batch_size)
            for batch in train_dataloader:
                goodModel.optimizer.zero_grad()
                loss = goodModel.training_step(batch)
                loss.backward()
                goodModel.optimizer.step()

            val_dataloader = DataLoader(dataset.val_dataset, batch_size)
            for batch in val_dataloader:
                goodModel.validation_step(batch, train=True)

            goodModel.epoch_end_callback(train=True)

            # logs the epoch results
            epoch_results = goodModel.epoch_results()
            for logger in self.loggers:
                logger(epoch_results)

    def add_babysitter(self, babysitter):
        self.babysitters.append(babysitter)

    def add_logger(self, logger):
        self.loggers.append(logger)

def change_optimizer_parameters(optimizer, hyperparameters):
    # Changes the parameters of the optimizer with the Trainer hyperparameters 
    for param in hyperparameters.keys():
        for param_group in optimizer.param_groups:
            if param in param_group:
                param_group[param] = hyperparameters[param]
