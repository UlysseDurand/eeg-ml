# Machine learning from Electroencephalogram

This is the internship subject of Ulysse Durand, at Mahindra University, Hyderabad, India, from 21/4/25 to 19/7/25, under the supervision of Dr Nidhi Goyal.

## Folders structure

- `Report` : The report of the project
- `Timetable.md` : My time management of the project
- `EEG-classification` : Classifying EEG signals from the task the participants were doing.
- `EEG2video` : Reconstructing a video seen by the participants from EEG data.

## Submodules

```
git submodule init
git submodule update
```

# EXPERIMENTS

https://docs.google.com/spreadsheets/d/1qsH8CgjU3HPRLuYScruy9Dfg6YXoPQrdLxwv0QKig3o/edit?gid=1956801427#gid=1956801427

## RUN PIPELINE

### Getting the elements


### Generate images from BrainVis

You need a lot of CPU RAM, 110 GiB seems to work

Be in BrainVis folder

- Create a `brainvis` conda environment with all the pip libraries
specified from `BrainVis/README.md`

- Run 
```
python create_path.py
```

- Add the clip folder from [here](https://github.com/openai/CLIP) to this folder

- Add stable diffusion checkpoint to `pretrained_model/v1-5-pruned-emaonly.ckpt`

- Add CVPR40 files to `data/EEG` (eeg_5_95_std.pth,
block_splits_by_image_all.pth, block_splits_by_image_single.pth)

- Add images from [here](https://drive.google.com/file/d/1k3Psdqhl0Saiol4Yauy6eCQK6_-Em05R/view?usp=drive_link) to `/data/image`

- Get the Checkpoints, from https://github.com/RomGai/BrainVis: 

Please leave your email address in the issue (we will respond as soon as possible), or contact us directly via email at hfu006@e.ntu.edu.sg (some emails might be missed). We will send you the checkpoint along with the usage instructions.

- Add them:
    - pretrained_model/clipfinetune_model_sub1.pkl
    - data/EEG_Feature_Label/test_image_names.pth
    - data/EEG_Feature_Label/test_pred.pth
    - data/EEG_Feature_Label/test_seqs.pth
    - data/EEG_Feature_labe/test_label.pth

- Run 
```
python imageBLIPtoCLIP.py
python imageLabeltoCLIP.py
```

- Have Cuda >=12 and activate conda environment
```
enroot create -n cuda12 [cuda 12 sqsh file]
enroot start --mount $HOME:$HOME cuda12
# cd to /home/you_username
./miniconda3/bin/conda activate brainvis
```

- Run
```
python cascade_diffusion.py
```

Copy`picture-gene` the content to 
