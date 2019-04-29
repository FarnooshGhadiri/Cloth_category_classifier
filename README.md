# Cloth_category_classifier

![img](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/attributes.jpg)

A PyTorch classifier for the DeepFashion dataset which performs category, attribute, and bounding box prediction. To quote the description from the [main webpage](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html):

[The DeepFashion dataset] *Category and Attribute Prediction Benchmark* evaluates the performance of clothing category and attribute prediction. This is a large subset of DeepFashion, containing massive descriptive clothing categories and attributes in the wild. It contains:
- 289,222 number of clothes images;
- 50 number of clothing categories, and 1,000 number of clothing attributes;
- Each image is annotated by bounding box and clothing type.

Note that we also use the data described here to augment our classifier with bounding box prediction. This is expected to help improve performance on the aforementioned category and attribute prediction tasks.

The architecture we used is not exactly like the one which was proposed in the original paper, though conceptually it incorporates the similar notions, such as having the predictions leverage both global and local features, the latter of which is achieved through the bounding box prediction part of the network.

<img src="https://user-images.githubusercontent.com/2417792/56816446-9086d900-6811-11e9-9afa-ce3787d50558.png" width=750 />


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

If you'd like to evaluate generalisation performance in post-hoc fashion, add `--mode=validate` or `--mode=test` to your training script to evaluate statistics on the validation and test set (by default, `--mode` is set to `train`).

### Testing the models

We have provided a pre-trained model which you can use by simply running:
```
bash download_pretrained.sh
```
This will download a model checkpoint and place it inside `results/pretrained_model`. (This model was trained with example script mentioned in the model training section. For more details about this model, e.g. statistics on how well it performs, please see the wiki section of this repository.)

Once this is done, you can use the script `test_on_image.py` to classify an image, for example:
```
python test_on_image.py \
--df_dir=<path_to_deepfashion_dir> \
--checkpoint=<path_to_checkpoint> \
--filename=<path_to_image> \
--out_file=<path_to_output>
```

Alternatively, you can process many images at once by using `--filenames` (instead of `--filename`) and `--out_folder` (instead of `--out_file`). Here is an example:

```
# process all images in Boxy_Pocket_Top
IMGS=`find '/tmp/beckhamc/deep_fashion/dataset/img/Boxy_Pocket_Top' -path '*.jpg' | tr '\n' ','`
python test_on_image.py \
--df_dir=/tmp/beckhamc/deep_fashion/ \
--checkpoint=results/pretrained_model/epoch_340.pth \
--filenames="${IMGS}" \
--out_folder=predictions/
```

To see the full list of options, simply run `python test_on_image.py --help`. Some extra options are `--top_k` (how many of the top attributes to retain per attribute category, which is the number of slices in the pie plots); and `--softmax_temp` (this is useful if the predictions are overly confident and you would like to give more weight to the classes with the smaller logits).


## Known issues

- The experimental setup is fundamentally flawed in the sense that many of the images contain more than one clothing item (e.g. skirt, blouse) but there is only one label for it (e.g. skirt). So if you are classifying an image of a person wearing more than one clothing item, the prediction will be confounded by features from the other clothing item. It turns out that each of the 50 categories belongs to one of three subcategories: topwear has the subcategory label '1' (e.g. shirts), bottomwear '2' (e.g. skirts), and both top and bottom '3' (e.g. dresses). Therefore, a better way to construct the classifier is to actually have three classification branches rather than one (I don't think they did this in the original paper).

## Troubleshooting

If you are experiencing any issues, please file a ticket in the Issues section.
