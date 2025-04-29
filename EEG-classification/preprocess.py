import torch

from ml_util.data_processing import normalize_tensor, apply_function_to_tensor
from layers.TimeFrequency import TimeFrequency

def preprocess_dataset(X : torch.Tensor, verbose=False):

	n = X.shape[2] # number of dots in signal
	total_time = 25 # in seconds
	fs = int(n / total_time) # sampling frequency of EEG
	hop_time = 0.5 # in seconds
	hop_length = int(fs * hop_time) # hop in number of dots

	X_normalized = normalize_tensor(X)
	timefreq = TimeFrequency(input_example=X[0], fs=fs, hop_length=hop_length)

	return apply_function_to_tensor(timefreq, X_normalized, batch_size=32, verbose=verbose)