from ml_util.data_module import DataModule
from ml_util.trainer import GoodClassificationModel

from torch.utils.data import DataLoader

def getTestResults(dataset: DataModule, model: GoodClassificationModel):
    model.epoch_begin_callback()    # Initializes the epoch variables
    test_dataloader = DataLoader(dataset.test_dataset, model.hyperparameters["batch_size"])
    for batch in test_dataloader:
        model.validation_step(batch, train=False)

    model.epoch_end_callback(train=False)
    res = model.epoch_results()
    return res