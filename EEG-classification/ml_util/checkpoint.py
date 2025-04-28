import os
import json
from typing import Callable
import torch

with open("config.json") as f:
    cfg = json.load(f)

def checkpoint(f: Callable[[], torch.Tensor], name: str, verbose=False):
	path = cfg["checkpoint_dir"]+"/"+name+".pt"

	if os.path.exists(path):
		if verbose: print(f"Checkpoint {name} found")
		return torch.load(path)
	else:
		if verbose: print(f"Checkpoint {name} not found")
		res = f()
		torch.save(res, path)
		return res 