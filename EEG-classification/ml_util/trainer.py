import torch
from torch import nn
from typing import Callable, Any
from torch.utils.data import DataLoader

from ml_util.data_module import DataModule, calculate_accuracy

class Trainer:
    def __init__(
        self, 
        model: nn.Module, 
        optimizer: torch.optim.Optimizer, 
        loss: Callable[[torch.Tensor], torch.Tensor], 
        hyperparameters={},
        device=torch.device("cpu"),
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.device = device
        self.hyperparameters = hyperparameters
        self.babysitting_callback = lambda itself, r: None
        self.result_callback = lambda itself, r: None

    def set_babysitting_callback(self, f: Callable[[object, int], None]):
        '''
        The result callback is called after each epoch
        It can be used to tweak the hyperparameters
        '''
        self.babysitting_callback = f

    def set_result_callback(self, f: Callable["...", None]):
        '''
        The result callback is called after each epoch
        It can be used to print the results or show a confusion matrix
        '''
        self.callback = f

    def change_epoch(self, epoch_result: int):
        # Called at each epoch change
        self.result_callback(self, epoch_result)
        self.babysitting_callback(self, epoch_result)

        # Changes the parameters of the optimizer with the Trainer hyperparameters 
        for param in self.hyperparameters.keys():
            for param_group in self.optimizer.param_groups:
                if param in param_group:
                    param_group[param] = self.hyperparameters[param]

    def train_epoch(
            self,
            dataset: DataModule,
    ):
        # DataLoaders split the whole dataset in batches
        train_loader = DataLoader(dataset.train_dataset, batch_size=self.hyperparameters["batch_size"])
        val_loader = DataLoader(dataset.val_dataset, batch_size=len(dataset.val_dataset))

        epoch_train_results = []
        self.model.train()
        for batch_X, batch_y in train_loader:
            # Transfer the train batch to, potentially, the GPU
            batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
            loss, pred = self.exec_batch(batch_X, batch_y, backprop=True)
            epoch_train_results.append((loss, pred, batch_y))
        
        epoch_val_results = [] 
        self.model.eval()
        for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
            loss, pred = self.exec_batch(batch_X, batch_y, backprop=False)
            epoch_val_results.append((loss, pred, batch_y))

        return epoch_train_results, epoch_val_results

    def exec_batch(
        self,
        batch_X: torch.tensor,
        batch_y: torch.tensor,
        backprop: bool 
    ):
        batch_output = self.model(batch_X)
        batch_loss = self.loss(batch_output, batch_y)
        batch_pred = batch_output.argmax(dim=1)

        if backprop: 
            batch_loss.backward()
            self.optimizer.step()

        return batch_loss * batch_X.shape[0], batch_pred

    def train(
            self,
            dataset: DataModule,
            num_epochs=10,
            verbose=False
    ):
        for epoch in range(num_epochs):
            train_epoch_result, val_epoch_result = self.train_epoch(dataset)

            train_loss, train_pred, train_real = parse_epoch_results(train_epoch_result) 
            val_loss, val_pred, val_real = parse_epoch_results(val_epoch_result)

            training_accuracy = calculate_accuracy(train_pred, train_real)
            val_accuracy = calculate_accuracy(val_pred, val_real)

            epoch_result = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_pred": train_pred,
                "train_real": train_real,
                "val_loss":val_loss,
                "val_pred":val_pred,
                "val_real":val_real
            }

            self.change_epoch(epoch_result)
            print(f"Epoch {epoch:4d}# Loss/train: {train_loss:2f},  Acc/train: {training_accuracy:2f}   |   Loss/val: {val_loss:2f},  Acc/val: {val_accuracy:2f}")

def parse_epoch_results(results):
    preds = []
    reals = []
    totalLoss = 0
    totalSize = 0
    for loss, pred, real in results:
        assert(len(pred) == len(real))
        totalLoss += loss * len(real)
        totalSize += len(real)

        preds += pred
        reals += real
    totalLoss /= totalSize
    return totalLoss, pred, real