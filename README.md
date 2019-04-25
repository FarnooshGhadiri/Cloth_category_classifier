# Cloth_category_classifier

![img](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/attributes.jpg)

A PyTorch classifier for the DeepFashion dataset which performs category, attribute, and bounding box prediction. To quote the description from the [main webpage](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html):

[The DeepFashion dataset] *Category and Attribute Prediction Benchmark* evaluates the performance of clothing category and attribute prediction. This is a large subset of DeepFashion, containing massive descriptive clothing categories and attributes in the wild. It contains:
- 289,222 number of clothes images;
- 50 number of clothing categories, and 1,000 number of clothing attributes;
- Each image is annotated by bounding box and clothing type.

Note that we also use the data described here to augment our classifier with bounding box prediction. This is expected to help improve performance on the aforementioned category and attribute prediction tasks.

## Setting up the project

### Cloning the repository:
`$ git clone https://github.com/FarnooshGhadiri/Cloth_category_classifier.git`

### Environment setup

1. Install Anaconda, if not already done, by following these instructions:
https://docs.anaconda.com/anaconda/install/linux/  

2. Create a conda environment using the `environment.yaml` file, to install the dependencies:  
`$ conda env create -f environment.yaml`

3. Activate the new conda environment:
`$ conda activate deepfashion`

4. (Optional) We also have implemented support for nvidia's mixed precision training library [Apex](https://github.com/NVIDIA/apex), which allows for float16 training (this can speed up training and provide memory savings on the GPU). Please see their README file for instructions on how to install this. Once it is installed, you can make use of it by appending `--fp16` to your training script.

### Getting the data

1. You can download the dataset from the Google Drive here:
https://drive.google.com/drive/folders/0B7EVK8r0v71pWGplNFhjc01NbzQ

2. Extract the dataset into some directory which is appropriate

## Running experiments

### Training the models

The script used to train a model is `train.py`. Here is the script we used to train our *best* model:

```
python train.py \
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

To understand what each keyword argument is, simply run `python train.py --help`.

During model training, after each epoch the script will append model statistics to `<experiment_name>/results.txt`, which details various things such as the losses and accuracies on both the training and validation set. This is useful for diagnosing model training.

### Testing the models

We have provided a pre-trained model which you can use here.

```
mkdir -p results/our_model
```
