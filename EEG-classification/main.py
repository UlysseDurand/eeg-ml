verbose = False

# All the imports

import torch
from torch.utils.data import TensorDataset
from torch import nn
import matplotlib.pyplot as plt

from load_dataset import labelList, get_data
from preprocess import preprocess_dataset
import bigmodel

from ml_util.eeg_util import plotEEG, plotTimeFreqEEG
from ml_util.checkpoint import checkpoint
from ml_util.data_module import DataModule, calculate_accuracy
from ml_util.trainer import Trainer
from ml_util.report_wandb import WandBReporter, print_stats

X_raw, y = get_data(verbose=True)

if verbose:
    plotEEG(X_raw[0], title="index 0")
    print(labelList[y[0]])

# Applies preprocessing
X = checkpoint(lambda : preprocess_dataset(X_raw, verbose=True), "preprocessed")
print(f"Input dataset shape: {X.shape}")

# Stores in a TensorDataset
dataset = DataModule(X, y)
if verbose:
    for label in range(4):
        sampleId = torch.where(y==label)[0][0]
        plt.figure()
        fig, axes = plotTimeFreqEEG(X[sampleId])
        #plt.imshow(X[sampleId][0])
        fig.suptitle(f"Sample {sampleId}, with label {labelList[y[sampleId]]}")
    plt.show()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("GPU" if torch.cuda.is_available() else "CPU")

hyperparams = {
    'lr': 4e-3,
    'weight_decay': 0,
    'batch_size': 16
}
model = bigmodel.model(X[0])
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams['lr'], weight_decay=hyperparams['weight_decay'])

trainer = Trainer(model, optimizer, loss_fn, hyperparams, device)

def babysitter(hyperparameters, epoch_results):
    hyperparameters['lr'] = 0.9 * hyperparameters['lr']

wandbreporter = WandBReporter(hyperparams)
def report(res):
    print_stats(res)
    wandbreporter.report(res)

trainer.set_report_callback(report)
trainer.set_babysitting_callback(babysitter)

trainer.train(dataset, 1000)