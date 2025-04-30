# EEG-classification

## Installation

1. Clone the repository:
```
git clone https://gitlab.com/ulysse_durand/eeg-ml#
cd eeg-ml/EEG-classification
```

2. Create a virtual environment:
```
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

3. Install the dependencies:
```
pip install -r requirements.txt
```

4. Download the Dataset [SAM 40](https://figshare.com/articles/dataset/SAM_40_Dataset_of_40_Subject_EEG_Recordings_to_Monitor_the_Induced-Stress_while_performing_Stroop_Color-Word_Test_Arithmetic_Task_and_Mirror_Image_Recognition_Task/14562090/1?file=27956376)

Extract the `filtered_data` folder here.

5. For Weights & Biases usage:

Put your API key in the .env file, it should look like this
```
WANDB_API_KEY=your_api_key
```

## Usage

run 
```
jupyter notebook
```

and open main.ipynb from your browser interface.
