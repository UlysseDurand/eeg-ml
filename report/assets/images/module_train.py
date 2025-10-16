from ml_util import *
from the_ml import dataset_load, MyModel, HistoryLogger

# From ml_util, uses params from arguments or default from config file
hyperparams = parse_hyperparams()

# Loads dataset (can eventually download from internet)
X_raw, y = checkpoint(dataset_load(), "raw_input") 

# Preprocesses the input
X = checkpoint(lambda : preprocess_dataset, X_raw, "preprocessed")

# X: data points, y: corresponding classes
dataset = DataModule(X, y, val_part=0.15, test_part=0.15)

device = ...
model = MyModel(device)

trainer = GoodTrainer() # From ml_util

# An example of training babysitting
def babysitter(goodModel):
    goodModel.hyperparameters['lr'] = 0.99 * goodModel.hyperparameters['lr']

trainer.add_babysitter(babysitter)

# Can add custom loggers (function executed after each new epoch trained)
historyLogger = HistoryLogger() # Stores the training history
trainer.add_logger(historyLogger)

# From ml_util, logs the results to Weight and Biases, Tensorboard also supported
trainer.add_logger(WandBLogger(hyperparams, labelList, model))

# Main training loop called with all the parameters
trainer.train(classification_model, dataset, hyperparams['epochs'])

# From ml_utils
res = getTestResults(dataset, model)
print(f" Test accuracy: {res['test_acc']}")

# From ml_utils
plot_loss_acc(historyLogger.history, fileNameBase="fig/history")

test_preds, test_labels = res["test_confusion"]
# From ml_utils
plot_confusion_matrix(test_preds, test_labels, labelList, title="Test Confusion Matrix", fileName="fig/test_confusion_matrix.png")