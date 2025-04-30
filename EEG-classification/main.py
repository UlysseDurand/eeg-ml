#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Define the verbose variable
verbose = False

# Check if we're running in a Jupyter Notebook (not compiled script)
try:
    __IPYTHON__
    verbose = True  # Set verbose to True if running in a Jupyter Notebook
except NameError:
    pass  # Do nothing if we're not in a notebook


# In[2]:


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
from ml_util.data_module import DataModule
from ml_util.trainer import GoodClassificationModel, GoodTrainer
from ml_util.logger import WandBReporter, print_stats


# In[3]:


print("Loading raw dataset")
X_raw, y = checkpoint(lambda: get_data(verbose=True), "raw_input")


# In[4]:


if verbose:
    plotEEG(X_raw[0], title="index 0")
    print(labelList[y[0]])


# In[5]:


# Applies preprocessing
print("Preprocessing input data")
X = checkpoint(lambda : preprocess_dataset(X_raw, verbose=True), "preprocessed")
print(f"Input dataset shape: {X.shape}")

# Stores in a TensorDataset
dataset = DataModule(X, y)

# Uncomment to try overfitting
# dataset = dataset.get_part(0.25)


# In[6]:


if verbose:
    for label in range(4):
        sampleId = torch.where(y==label)[0][0]
        plt.figure()
        fig, axes = plotTimeFreqEEG(X[sampleId])
        #plt.imshow(X[sampleId][0])
        fig.suptitle(f"Sample {sampleId}, with label {labelList[y[sampleId]]}")
    plt.show()


# In[7]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("GPU" if torch.cuda.is_available() else "CPU")

hyperparams = {
    'lr': 1e-3,
    'weight_decay': 5e-3,
    'batch_size': 16
}
model = bigmodel.model(X[0])
loss_fn = nn.CrossEntropyLoss()

classification_model = GoodClassificationModel(model, loss_fn, hyperparams, device, 4)
trainer = GoodTrainer()

def babysitter(goodModel):
    if (goodModel.epoch %  10 == 0):
        goodModel.hyperparameters['lr'] = 0.8 * goodModel.hyperparameters['lr']

trainer.add_babysitter(babysitter)
trainer.add_logger(print_stats)
trainer.add_logger(WandBReporter(hyperparams))


# In[ ]:


trainer.train(classification_model, dataset, 1000)

