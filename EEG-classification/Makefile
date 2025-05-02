.PHONY: venv run clean_checkpoint clean install notebook 

run: main.py dataset/filtered_data
	. venv/bin/activate && python main.py

dataset/filtered_data: dataset/Data.rar
	cd dataset && \
	tar -xf Data.rar Data/filtered_data && \
	rm -rf filtered_data && \
	mv -f Data/filtered_data ./ && \
	rm -rf Data

dataset/Data.rar:
	cd dataset && python dl_dataset_1.py

main.py:
	. venv/bin/activate && jupyter nbconvert --to script main.ipynb

notebook: dataset/filtered_data
	. venv/bin/activate && jupyter notebook

install:
	python -m venv venv
	. venv/bin/activate && pip install -r requirements.txt

clean_checkpoint:
	rm -f -- checkpoints/*

clean:
	git clean -fdX
