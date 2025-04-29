import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.data import TensorDataset
import numpy as np
from typing import Callable

class DataModule():
    '''
    MyDataModule.whole_dataset is the whole dataset
    MyDataModule.train_dataset is the training dataset 
    MyDataModule.test_dataset is the testing dataset
    '''

    def __init__(self, X: torch.Tensor, y: torch.Tensor, batch_size=32, val_part=0.15, test_part=0.15):
        super().__init__()
        self.batch_size = batch_size
        self.whole_dataset = TensorDataset(X, y)
        self.val_part = val_part
        self.test_part = test_part
        self.split_train_val_test()

    def split_train_val_test(self):
        self.val_len = int(self.val_part * len(self.whole_dataset))
        self.test_len = int(self.test_part * len(self.whole_dataset))
        self.train_len = len(self.whole_dataset) - self.test_len - self.val_len

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(self.whole_dataset, [self.train_len, self.val_len, self.test_len])

    def evaluate_classification_model(self, model : Callable[[torch.Tensor], int], verbose=False):
        '''
        Evaluating the model on the test dataset
        '''
        predicted_Y = []
        real_Y = []
        for (x, y) in self.test_dataset:
            predicted_Y += model(x).argmax(dim=1)
            real_Y += y

        test_acc = calculate_accuracy(predicted_Y, real_Y)
        print()
        return predicted_Y

def calculate_accuracy(predicted, real) -> float:
    assert len(predicted) == len(real)
    good = 0
    for i in range(len(predicted)):
        if predicted[i] == real[i]:
            good += 1
    return good / len(predicted)

def calculate_confusion_matrix(preds, real, nblabels: int):
    res = np.zeros((nblabels, nblabels))
    for i in range(len(preds)):
        thepred = preds[i]
        thereal = real[i]
        res[thereal][thepred] += 1
    return res

def calculate_epoch(res):
    train_pred, train_real = res["train_pred"], res["train_real"]
    val_pred, val_real = res["val_pred"], res["val_real"]
    nblabels = res["nblabels"]

    res["train_acc"] = calculate_accuracy(train_pred, train_real)
    res["val_acc"] = calculate_accuracy(val_pred, val_real)

    res["train_confusion"] = calculate_confusion_matrix(train_pred, train_real, nblabels)
    res["val_confusion"] = calculate_confusion_matrix(val_pred, val_real, nblabels)