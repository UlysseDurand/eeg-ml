# EEG-classification

## Installation

1. Clone the repository:
```
git clone https://gitlab.com/ulysse_durand/eeg-ml#
cd eeg-ml/EEG-classification
```

2. Create the environment & install the pip dependencies:
```
make install
```

3. Download the dataset
```
make dataset/Data.rar
```

4. Extract the `filtered_data` folder in `dataset/filtered_data`.

5. For Weights & Biases usage:

Put your API key in the `.env` file, it should look like this
```
WANDB_API_KEY=your_api_key
```

## Run notebook

```
make notebook
```

and open main.ipynb from your browser interface.

## Run with console

This converts the `main.ipynb` notebook to `main.py` and runs it.

```
make run
```

## Make the report

This converts the `main.ipynb` notebook to `main.pdf` after running the notebook.

```
make report
```
