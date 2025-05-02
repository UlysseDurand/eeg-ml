from tqdm import tqdm
from scipy.io import loadmat
import numpy as np
import os
import torch

labelList = ['Arithmetic', 'Mirror_image', 'Relax', 'Stroop']
nbSubjects = 40
nbTrials = 3

def get_sample(task, subjectId, trialNb, folder='dataset/filtered_data'):
    '''Retrieves one sample from the data folder''' 
    assert (task in labelList)
    assert (0 < subjectId <= nbSubjects)
    assert (0 < trialNb <= nbTrials)

    return loadmat(folder+'/'+task+'_sub_'+str(subjectId)+'_trial'+str(trialNb)+'.mat')['Clean_data']

def get_data(data_folder='dataset/filtered_data', verbose=False, device=torch.device("cpu")) -> tuple[torch.Tensor, torch.Tensor]:

    if not(os.path.isdir(data_folder)): raise FileNotFoundError(f"Folder '{data_folder}' not found, the dataset is the `filtered_data` folder from https://figshare.com/articles/dataset/SAM_40_Dataset_of_40_Subject_EEG_Recordings_to_Monitor_the_Induced-Stress_while_performing_Stroop_Color-Word_Test_Arithmetic_Task_and_Mirror_Image_Recognition_Task/14562090/1?file=27956376")

    if verbose: print("Loading dataset")

    total_steps = len(labelList) * nbSubjects * nbTrials
    if verbose:
        pbar = tqdm(total=total_steps, bar_format='{desc:<30} | {n}/{total} [{bar}]')
    else: pbar=None
        
    dataset_array = []
    for task in labelList:
        for subjectId in range(1,40+1):
            for trialNb in range(1,3+1):
                sample_x = get_sample(task, subjectId, trialNb, data_folder)
                label = labelList.index(task)
                dataset_array.append((sample_x, label))
                
                if verbose:
                    pbar.set_description(f"Task: {task:<13}, Subject: {subjectId:2d}, Trial: {trialNb:1d}")
                    pbar.update(1)

    X_array, Y_array = zip(*dataset_array) 

    X = torch.tensor(np.array(X_array)).float()
    y = torch.tensor(np.array(Y_array))

    assert (X.shape[0] == y.shape[0])

    if verbose: print(f"Dataset loaded, X : {X.shape}")

    return X, y
