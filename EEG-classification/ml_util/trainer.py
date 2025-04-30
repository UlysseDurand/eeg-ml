import torch
from torch import nn
from typing import Callable, List, Tuple, Dict, Any
from torch.utils.data import DataLoader
from torchmetrics.classification import Accuracy

from ml_util.data_module import DataModule, evaluate_classification_model

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

        self.train_acc  = Accuracy("multiclass", num_classes=nblabels)
        self.val_acc    = Accuracy("multiclass", num_classes=nblabels)
        self.epoch_log = {}
        self.epoch = 0

    def epoch_begin_callback(self):
        self.train_acc.reset()
        self.val_acc.reset()
        self.epoch_log = {}

    def training_step(self, batch):
        # Returns what needs to be minimized
        self.model.train()
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        scores = self.model(x)
        loss = self.loss(scores, y)
        self.train_acc.update(scores, y)
        self.log("train_loss", (loss, len(batch)))
        return loss

    def validation_step(self, batch):
        self.model.eval()
        with torch.no_grad():
            x, y = batch
            x, y = x.to(self.device), y.to(self.device)
            scores = self.model(x)
            loss = self.loss(scores, y)
            self.val_acc.update(scores, y)
            self.log("val_loss", (loss, len(batch)))

    def epoch_end_callback(self):
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
        res["train_loss"] = self.calculate_loss("train_loss")
        res["val_loss"] = self.calculate_loss("val_loss")
        res["train_acc"] = self.train_acc.compute()
        res["val_acc"] = self.val_acc.compute()
        return res

    def calculate_loss(self, key):
        res = None
        if key in self.epoch_log:
            res = 0
            total = 0
            for a, b in self.epoch_log[key]:
                res += a * b
                total = b
            return res / total
        else:
            raise RuntimeError(f"{key} loss not logged during epoch calculation")
    
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
                goodModel.validation_step(batch)

            goodModel.epoch_end_callback()

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