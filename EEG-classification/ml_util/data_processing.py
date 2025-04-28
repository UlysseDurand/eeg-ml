import torch

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