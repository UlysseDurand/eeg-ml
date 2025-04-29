import wandb
import os
import json
import numpy as np
from ml_util.data_module import calculate_epoch

with open("config.json") as f:
    cfg = json.load(f)


class WandBReporter():
    def __init__(self, hyperparameters):
        self.run = wandb.init(
            project=cfg.get("project", "project"),
            config=hyperparameters
        )

    def report(self, res):
        keys_to_send = ["train_loss", "train_acc", "val_loss", "val_acc"]
        to_send = {key : res[key] for key in keys_to_send}
        self.run.log(to_send)

def print_stats(res):
    calculate_epoch(res)
    print(f"Epoch:{res["epoch"]:>5}    | tl: {res["train_loss"]:2f}, ta: {res["train_acc"]:2f}    |    vl: {res["val_loss"]:2f}, va: {res["val_acc"]:2f}")
