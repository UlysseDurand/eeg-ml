execute: makescript run

makescript:
	jupyter nbconvert --to script main.ipynb

run:
	python main.py

notebook:
	jupyter notebook

no_checkpoint:
	rm -f -- checkpoints/*

clean:
	git clean -fdX
