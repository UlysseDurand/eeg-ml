import torch
from tqdm import tqdm
from typing import Callable, Any
from torch.utils.data import DataLoader

def normalize_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """
    Normalizes a PyTorch tensor by subtracting the mean and dividing by the standard deviation.
    :param tensor: PyTorch tensor to normalize
    :return: Normalized PyTorch tensor
    """
    mean = tensor.mean()
    std = tensor.std()
    normalized_tensor = (tensor - mean) / std
    return normalized_tensor

def apply_function_to_tensor(
        f: Callable[[torch.Tensor], Any], 
        X: torch.tensor, 
        reconstruct=lambda ress : torch.cat(ress,dim=0), 
        batch_size=32, verbose=False
    ):
    '''
    f is a function from a tensor to a tensor, it is meant to be used with the whole dataset
    This function makes it applied to batches that are then concatenated
    '''
    if verbose:
        the_tqdm = tqdm
    else:
        the_tqdm = lambda x, *args: x

    dataloader = DataLoader(X, batch_size=batch_size, shuffle=False)
    all_outputs = []

    with torch.no_grad():
        for inputs in the_tqdm(dataloader, desc="Batch function apply"):
            out = f(inputs)
            all_outputs.append(out.cpu())
        return reconstruct(all_outputs)