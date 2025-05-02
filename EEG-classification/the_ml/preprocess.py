import torch

from ml_util.data_processing import normalize_tensor, apply_function_to_tensor
from the_ml.TimeFrequency import TimeFrequency

def getbands(bandsparam, fs):
	if (bandsparam == "five"):
		bands = [
			(0.5,4),
			(4,8),
			(8,12),
			(12,30),
			(30,63.9)
		]

		deltafreq = 4
		hop_length = int(fs / deltafreq)

		return bands, hop_length
	else:
		hop_time = bandsparam

		hop_length = int(fs * hop_time) # hop in number of dots
		deltafreq = fs / hop_length

		nbbands = int((fs / 2) / deltafreq)
		bands = []
		for i in range(nbbands):
			bands.append((i*deltafreq, (i+1)*deltafreq))

		return bands, hop_length


def preprocess_dataset(X : torch.Tensor, bandsparam, verbose=False):
	# bandsparam is either "five" for the usual bands, either a hop time in
	# seconds in the first case the hop time is calculated, in the second case
	# the number of bands will be calculated

	n = X.shape[2] # number of dots in signal
	total_time = 25 # in seconds
	fs = int(n / total_time) # sampling frequency of EEG

	bands, hop_length = getbands(bandsparam, fs)

	X_normalized = normalize_tensor(X)
	timefreq = TimeFrequency(X[0], bands, fs=fs, hop_length=hop_length)

	return apply_function_to_tensor(timefreq, X_normalized, batch_size=32, verbose=verbose)
