import torch
from typing import Callable
import warnings
from tqdm import tqdm

def showprogress(f: Callable[[torch.Tensor], torch.Tensor], X: torch.Tensor):
	'''
	This function applies f to every element of X, showing a progress bar.
	This is slower than X.apply_(f)
	'''

	warnings.warn("showprogress is slower than tensor.apply_()")

	res = torch.empty_like(X)
	pbar = tqdm(total=X.numel(), desc="Preprocessing step", position=0)
	for i, sample in enumerate(X):
		res[i] = f(sample)
		pbar.update(1)	
	return res