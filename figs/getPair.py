import os
import sys
import importlib.util
import matplotlib.pyplot as plt

sys.path.append('.')
from DreamDiffusion.code.dataset import create_EEG_dataset

spec = importlib.util.spec_from_file_location('plotEEG', "EEG-classification/ml_util/eeg_util.py")
module = importlib.util.module_from_spec(spec)
sys.modules['plotEEG'] = module
spec.loader.exec_module(module)

def main():
    dataPath = "DreamDiffusion/datasets/eeg_5_95_std.pth"
    imageNetPath = "DreamDiffusion/datasets/imageNet_images/"
    splitsPath = "DreamDiffusion/datasets/block_splits_by_image_single.pth"

    print("Loading the dataset...")
    dataset_train, dataset_test = create_EEG_dataset(
        eeg_signals_path=dataPath,
        splits_path=splitsPath,
        imagenet_path=imageNetPath
    )

    print(f"Dataset loaded of length {len(dataset_train) + len(dataset_test)}")
    print(dataset_train[0]['eeg'].shape)
    module.plotEEG(dataset_train[20]['eeg'], range(64), scaler=10)
    # plt.imshow(dataset_train[20]['image'])
    plt.show()



if __name__ == '__main__':
    main()