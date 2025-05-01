.PHONY: venv run clean_checkpoint clean install notebook 

run: main.py
	. venv/bin/activate && python main.py

main.py:
	. venv/bin/activate && jupyter nbconvert --to script main.ipynb

notebook:
	. venv/bin/activate && jupyter notebook

install:
	python -m venv venv
	. venv/bin/activate && pip install -r requirements.txt

clean_checkpoint:
	rm -f -- checkpoints/*

clean:
	git clean -fdX
