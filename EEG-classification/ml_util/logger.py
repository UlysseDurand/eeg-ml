import wandb
import json
import os
from dotenv import load_dotenv

load_dotenv()

with open("config.json") as f:
    cfg = json.load(f)

class HistoryLogger():
    def __init__(self):
        self.history = {
        }

    def __call__(self, res):
        for key, value in res.items():
            if key in self.history:
                self.history[key].append(value)
            else:
                self.history[key] = [value]


class WandBLogger():
    def __init__(self, hyperparameters, labelList, model):
        self.labelList = labelList
        wandb.login(key=os.environ.get("WANDB_API_KEY"))
        self.run = wandb.init(
            project=cfg.get("project", "project"),
            config=hyperparameters
        )
        self.run.watch(model)


    def __call__(self, res):
        for part in ["train", "val"]:
            for metric in ["loss", "acc"]:
                if (f"{part}_{metric}" in res):
                    self.run.log({
                        f"{part}/{metric}": res[f"{part}_{metric}"]
                    })
            if f"{part}_confusion" in res and part=="val":
                preds, labels = res["val_confusion"]
                preds = preds.cpu().numpy().tolist()
                labels = labels.cpu().numpy().tolist()
                self.run.log({
                    f"{part}/confusion": wandb.plot.confusion_matrix(preds=preds, y_true=labels, class_names=self.labelList)
                })

def print_stats(res, print_every=1):
    if (res['epoch'] % print_every == 0):
        print(f"Epoch:{res['epoch']:>5}    | tl: {res['train_loss']:2f}, ta: {res['train_acc']:2f}    |    vl: {res['val_loss']:2f}, va: {res['val_acc']:2f}")
