# Cloth_category_classifier

A PyTorch classifier for the DeepFashion dataset which performs category, attribute, and bounding box prediction.

## Setting up the project

### Cloning the repository:
`$ git clone https://github.com/simonguiroy/RecognizingViolentActions.git`

### Environment setup

1. Install Anaconda, if not already done, by following these instructions:
https://docs.anaconda.com/anaconda/install/linux/  

2. Create a conda environment using the `environment.yaml` file, to install the dependencies:  
`$ conda env create -f environment.yaml`

3. Activate the new conda environment:
`$ conda activate RecognizingViolentActions`

### Getting the data

1. You can download the dataset from the Google Drive here:
https://drive.google.com/drive/folders/0B7EVK8r0v71pWGplNFhjc01NbzQ

2. Extract the dataset into some directory which is appropriate

## Running experiments

### Training the models

The script used to train a model is `attr_classification.py`. Here is the main script we used to train our best model:

```
python attr_classification.py \
--img_size=256 \
--crop_size=224 \
--num_workers=8 \
--data_dir=<path_to_data_dir> \
--batch_size=32 \
--name=<experiment_name> \
--reduce_sum \
--beta=0.001 \
--resume=auto \
--epochs=200 \
--data_aug \
--lr=2e-4
```

To understand what each keyword argument is, simply run `python attr_classification.py --help`. For the sake of convenience, the table below describes what each of the arguments does:


