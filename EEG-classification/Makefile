.PHONY: venv run clean_checkpoint clean install notebook 

NVIDIA_GPU_CHECK := $(shell nvidia-smi > /dev/null 2>&1 && echo 1 || echo 0)

run: main.py dataset/filtered_data
	. venv/bin/activate && python main.py

dataset/Data.rar:
	@echo "Downloading dataset..."
	@cd dataset && python dl_dataset.py
	@echo "Now please extract filtered_data from dataset/Data.rar to dataset/filtered_data"

main.py: main.ipynb
	@echo "Converting notebook to script..."
	@. venv/bin/activate && jupyter nbconvert --to script main.ipynb

notebook: dataset/filtered_data
	@. venv/bin/activate && jupyter notebook

install:
	@echo "Installation of virtual enviroment"
	@python -m venv venv
	@echo "Checking for NVIDIA GPU..."
	@if [ "$(NVIDIA_GPU_CHECK)" = "1" ]; then \
		echo "NVIDIA GPU detected. Installing CUDA version of PyTorch..."; \
		venv/bin/pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117; \
	else \
		echo "No NVIDIA GPU detected. Installing CPU-only version of PyTorch..."; \
		venv/bin/pip install torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1; \
	fi
	@echo "Installing requirements"
	@. venv/bin/activate && venv/bin/pip install -r requirements.txt

clean_checkpoint:
	rm -f -- checkpoints/*

clean:
	git clean -fdX

main.pdf: main.py
	. venv/bin/activate && jupyter nbconvert --to pdf --execute main.ipynb --log-level=INFO
