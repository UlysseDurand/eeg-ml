import wandb
import json
import os
from dotenv import load_dotenv

from ml_util.data_module import calculate_epoch
load_dotenv()

with open("config.json") as f:
    cfg = json.load(f)

class WandBReporter():
    def __init__(self, hyperparameters):
        wandb.login(key=os.environ.get("WANDB_API_KEY"))
        self.run = wandb.init(
            project=cfg.get("project", "project"),
            config=hyperparameters
        )

    def __call__(self, res):
        self.run.log({
            "train/loss": res["train_loss"],
            "val/loss": res["val_loss"],
            "train/acc": res["train_acc"],
            "val/acc" : res["val_acc"]
        })

def print_stats(res):
    print(f"Epoch:{res["epoch"]:>5}    | tl: {res["train_loss"]:2f}, ta: {res["train_acc"]:2f}    |    vl: {res["val_loss"]:2f}, va: {res["val_acc"]:2f}")
