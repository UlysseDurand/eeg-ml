import torch
from torch.utils.data import random_split
from torch.utils.data import TensorDataset, Dataset
import numpy as np
from typing import Callable

class DataModule():
    '''
    MyDataModule.whole_dataset is the whole dataset
    MyDataModule.train_dataset is the training dataset 
    MyDataModule.test_dataset is the testing dataset
    '''

    def __init__(self, X: torch.Tensor, y: torch.Tensor, val_part=0.15, test_part=0.15):
        self.whole_dataset = TensorDataset(X, y)
        self.val_part = val_part
        self.test_part = test_part
        self.split_train_val_test()

    def split_train_val_test(self):
        self.val_len = int(self.val_part * len(self.whole_dataset))
        self.test_len = int(self.test_part * len(self.whole_dataset))
        self.train_len = len(self.whole_dataset) - self.test_len - self.val_len

        X, y = self.whole_dataset.tensors

        train_sub, val_sub, test_sub = random_split(self.whole_dataset, [self.train_len, self.val_len, self.test_len])
        self.train_dataset = TensorDataset(X[train_sub.indices], y[train_sub.indices])
        self.val_dataset = TensorDataset(X[val_sub.indices], y[val_sub.indices])
        self.test_dataset = TensorDataset(X[test_sub.indices], y[test_sub.indices])

    def get_part(self, part):
        indices = get_part_indices(self.whole_dataset, part)
        X, y = self.whole_dataset.tensors
        newX = X[indices]
        newy = y[indices]
        return DataModule(newX, newy, self.val_part, self.test_part)

def get_part_indices(dataset: Dataset, part):
    # Returns a random smaller dataset of size part*len(self.whole_dataset)
    subset, _ = random_split(dataset, [part, 1-part])
    return subset.indices

def calculate_accuracy(predicted, real) -> float:
    assert len(predicted) == len(real)
    good = 0
    for i in range(len(predicted)):
        if predicted[i] == real[i]:
            good += 1
    return good / len(predicted)