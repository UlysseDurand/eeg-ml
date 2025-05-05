#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# All the imports

import torch
from torch.utils.data import TensorDataset
from torch import nn
import matplotlib.pyplot as plt
import argparse

from the_ml.load_dataset import labelList, get_data
from the_ml.preprocess import preprocess_dataset
from the_ml import bigmodel

from ml_util.eeg_util import plotEEG, plotTimeFreqEEG
from ml_util.checkpoint import checkpoint
from ml_util.data_module import DataModule
from ml_util.trainer import GoodClassificationModel, GoodTrainer
from ml_util.logger import WandBLogger, print_stats
from ml_util.reporter import getTestResults
from ml_util.plot import plot_confusion_matrix

# To make it reproducible
torch.manual_seed(42)


# In[ ]:


parser = argparse.ArgumentParser()

parser.add_argument('--bands', default="five", help='number of bands for time-frequency spectrogram ("five") for usual bands')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='Weight decay (L2 regularization)')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
parser.add_argument('--verbose', type=bool, default=True, help='Make figures')

args = parser.parse_args()
hyperparams = {}

print("Arguments: ")
verbose = parser.verbose

for attr, value in vars(args).items():
    print(f'{attr}: {value}')
    hyperparams[attr] = value


# In[3]:


print("Loading raw dataset")
X_raw, y = checkpoint(lambda: get_data(verbose=True), "raw_input")
print(f"Raw input dataset size: {X_raw.shape}")


# # The SAM40 Dataset
# 
# It consists of electroencephalograms (EEG) of people while they perform tasks.
# 
# You can find the dataset [here](https://figshare.com/articles/dataset/SAM_40_Dataset_of_40_Subject_EEG_Recordings_to_Monitor_the_Induced-Stress_while_performing_Stroop_Color-Word_Test_Arithmetic_Task_and_Mirror_Image_Recognition_Task/14562090)
# 
# We extract the dataset to obtain 480 samples (120 for each task among Stroop, Relax, Mirror_Image, and Arithmetic).
# Each sample has the record of the 32 electrods of the EEG over 25 seconds at 128Hz (3200 points)

# In[ ]:


if verbose:
    plotEEG(X_raw[0], title="index 0")
    print(labelList[y[0]])
    plt.savefig(f"fig/EEG_sample_example.png", dpi=300, bbox_inches='tight')


# ## Preprocessing
# 
# We don't want to feed the raw inputs to our model, we will extract features from it.
# 
# The approach used was to use a time-frequency spectrogram with the alpha, beta, gamma, delta and theta bands (those are intervals in the frequency domain).
# 
# After the preprocessing step we have a 32 channels 5x101 image for each sample. 5 corresponds to the 5 frequency intervals and 101 are the time intervals.

# In[5]:


hyperparams ={"bands_param": "five"} # hop time in seconds

# Applies preprocessing
print("Preprocessing input data")
X = checkpoint(lambda : preprocess_dataset(X_raw, hyperparams["bands_param"], verbose=True), "preprocessed")
print(f"Preprocessed inpu dataset shape: {X.shape}")

# Stores in a TensorDataset
dataset = DataModule(X, y, val_part=0.15, test_part=0.15)

# Uncomment to reduce the size of the dataset
# dataset = dataset.get_part(0.1)


# In[ ]:


if verbose:
    for label in range(4):
        sampleId = torch.where(y==label)[0][0]
        plt.figure()
        fig, axes = plotTimeFreqEEG(X[sampleId])
        fig.suptitle(f"Sample {sampleId}, with label {labelList[y[sampleId]]}")
        plt.savefig(f"fig/{labelList[y[sampleId]]}_timefreq.png", dpi=300, bbox_inches='tight')
    plt.show()


# ## The model
# 
# Here are the layers of the model: 
# - BatchNorm2d
# 
# - Conv2d          
# 32 to 128 channels, kernel_size of 5, stride 1, padding 2
# 
# - Dropout2d       
# 0.3 probability of channel dropout
# 
# - BatchNorm2d
# 
# - ReLU
# 
# - Conv2d          
# 128 to 64 channels, kernel_size of 5, stride 1, padding 2
# 
# - Dropout2d       
# 0.2 probability of channel dropout
# 
# - BatchNorm2d
# 
# - ReLU
# 
# - Conv2d          
# 64 to 32 channels, kernel_size of 5, stride 1, padding 2
# 
# - Dropout2d       
# 0.2 probability of channel dropout
# 
# - BatchNorm2d
# 
# - ReLU
# 
# - Flatten
# 
# - Fully-Connected 
# 32*101*5 to 128 layers
# 
# - Dropout         
# 0.5 probability of neuron dropout
# 
# - BatchNorm
# 
# - ReLU
# 
# - Fully-Connected 
# 128 to 4 layers
# 
# - SoftMax loss
# 
# 
# ## The training
# 
# - Adam optimizer was used
# - L2 regularization added to the loss
# - Every epoch, the learning rate is reduced by 1%

# In[ ]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("GPU" if torch.cuda.is_available() else "CPU")

model = bigmodel.model(X[0], hyperparams)
loss_fn = nn.CrossEntropyLoss()

classification_model = GoodClassificationModel(model, loss_fn, hyperparams, device, 4)
trainer = GoodTrainer()

def babysitter(goodModel):
    goodModel.hyperparameters['lr'] = 0.99 * goodModel.hyperparameters['lr']

trainer.add_babysitter(babysitter)
trainer.add_logger(lambda res: print_stats(res, print_every=100))
trainer.add_logger(WandBLogger(hyperparams, labelList, model))


# In[ ]:


trainer.train(classification_model, dataset, hyperparams['epochs'])


# # Results

# In[ ]:


res = (getTestResults(dataset, classification_model))
print(f" Test accuracy: {res['test_acc']}")
test_preds, test_labels = res["test_confusion"]
if verbose:
    plot_confusion_matrix(test_preds, test_labels, labelList)
    plt.savefig("fig/test_confusion_matrix.png", dpi=300, bbox_inches='tight')

