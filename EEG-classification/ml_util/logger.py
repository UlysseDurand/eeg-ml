import wandb
import json
import os
from dotenv import load_dotenv

load_dotenv()

with open("config.json") as f:
    cfg = json.load(f)

class WandBReporter():
    def __init__(self, hyperparameters, labelList, model):
        self.labelList = labelList
        wandb.login(key=os.environ.get("WANDB_API_KEY"))
        self.run = wandb.init(
            project=cfg.get("project", "project"),
            config=hyperparameters
        )
        self.run.watch(model)


    def __call__(self, res):
        val_preds, val_labels = res["val_confusion"]
        val_preds = val_preds.cpu().numpy().tolist()
        val_labels = val_labels.cpu().numpy().tolist()
        self.run.log({
            "train/loss": res["train_loss"],
            "val/loss": res["val_loss"],
            "train/acc": res["train_acc"],
            "val/acc" : res["val_acc"]
        })
        # self.run.log(
        #     "val/conf_mat": wandb.plot.confusion_matrix(preds=val_preds, y_true=val_labels, class_names=self.labelList)
        # )

def print_stats(res):
    print(f"Epoch:{res['epoch']:>5}    | tl: {res['train_loss']:2f}, ta: {res['train_acc']:2f}    |    vl: {res['val_loss']:2f}, va: {res['val_acc']:2f}")
