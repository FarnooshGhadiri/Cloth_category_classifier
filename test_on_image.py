# matplotlib stuff needs to be imported
# first
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import patches
# other imports
import numpy as np
import torch
import os
from torchvision.transforms import (Compose,
                                    Resize, 
                                    Normalize, 
                                    ToTensor,
                                    Lambda,
                                    TenCrop)
from PIL import Image
from models.model import FashionResnet
from data.dataset_fashion import get_attr_names_and_types
from data import input
from util import util
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--df_dir', type=str, required=True,
                        help="""Directory to DeepFashion directory which contains
                        the 'Anno' folder.""")
    # Either you specify a filename or a comma-separated list of
    # filenames.
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--filename', type=str,
                       help="Filename of the image you wish to classify.")
    group.add_argument('--filenames', type=str,
                       help="Filenames (space-separated) of the images you wish to classify.")
    # If a filename is provided, you must use `out_file`, otherwise,
    # you use `out_folder`.
    group2 = parser.add_mutually_exclusive_group(required=True)
    group2.add_argument('--out_file', type=str,
                        help="Output file")
    group2.add_argument('--out_folder', type=str,
                        help="Output folder")
    parser.add_argument('--resnet_type', default='resnet18',
                        choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'],
                        help="""What resnet type to use (must be compatible with the
                        specified model checkpoint""")
    parser.add_argument('--checkpoint', type=str, required=True,
                        help="Checkpoint file for model.")
    parser.add_argument('--softmax_temp', type=float, default=1.,
                        help="Softmax temperature")
    parser.add_argument('--top_k', type=int, default=3,
                        help="""How many predicted attributes (per attribute category)
                        should we show?""")
    args = parser.parse_args()
    return args

args = parse_args()

if args.filename is not None and args.out_file is None:
    raise Exception("If you specify `filename` you must use `out_file`")
if args.filenames is not None and args.out_folder is None:
    raise Exception("If you specify `filenames` you must use `out_folder`")

cat_names = input.get_ctg_name(args.df_dir)[0]
# BUG: we accidentally didn't use zero-indexed labels for the
# category classification part. So what we need to do is pad
# the start of `cat_names` with a no-class, and then keep it
# length 50 by removing one label from the end (which is ok,
# since it turns out there are no labels in the dataset for
# that last label).
cat_names = ['n/a'] + cat_names[0:-1]
attr_names, attr_types = get_attr_names_and_types(args.df_dir)
attr_names = np.asarray(attr_names)
attr_types = torch.FloatTensor(np.asarray(attr_types))
len(attr_names), len(attr_types)
attr_type_names = ['texture', 'fabric', 'shape', 'part', 'style']

def load_image(filename, root="", ensemble=False):
    """Load an image.

    :param filename: path to image
    :param root: can be specified, if filename is a relative path
    :param ensemble: if `True`, perform ten crops and return that
      instead.
    :returns: an image of dimension (1,3,224,224) if `ensemble`
      is `False`, otherwise (10,3,224,224).
    """
    transform_list = []
    if ensemble:
        norm = Normalize( (0.5, 0.5, 0.5), (0.5, 0.5, 0.5) )
        transform_list.append(Resize( (256, 256)))
        transform_list.append(TenCrop(224))
        transform_list.append(
            Lambda(lambda crops: torch.stack([ToTensor()(crop) for crop in crops]))
        )
        transform_list.append(
            Lambda(lambda crops: torch.stack([norm(crop) for crop in crops]))
        )
    else:
        transform_list.append(Resize( (224, 224)))
        transform_list.append(ToTensor())
        transform_list.append(Normalize( (0.5, 0.5, 0.5), (0.5, 0.5, 0.5) ))
    transformer = Compose(transform_list)
    filepath = "%s/%s" % (root, filename)
    img = Image.open(filepath)
    img = img.convert("RGB")
    img = transformer(img)
    if ensemble:
        return img
    else:
        return img.unsqueeze(0)

def plot_classification(net, filename, out_file, softmax_temp=1.0, ensemble=True, top_k=3):
    """Produce an image detailing the category and attribute
    classification of the given input.

    :param net: network
    :param filename: path to the image to be classified
    :param softmax_temp: softmax temperature for category
      classification.
    :param ensemble: perform 10 random crops and average
      the predictions?
    :param top_k: for attribute classification, how many
      of the top-performing attributes we should retain?
    :returns: 
    """

    util.mkdir(os.path.dirname(out_file))
    
    with torch.no_grad():
        img = load_image(filename=filename, root="", ensemble=ensemble)
        out_cls, out_bin, out_bbox = net(img)
        
        out_cls = out_cls.mean(dim=0, keepdim=True)
        out_bin = out_bin.mean(dim=0, keepdim=True)
        
        if ensemble:
            out_bbox = out_bbox[4:5] # the central crop?
            img = img[4]
        else:
            img = img[0]

    # Set up the grid.
    fig = plt.figure(figsize=(20, 9))
    grid = plt.GridSpec(2, 5, wspace=0.4, hspace=0.8, )
    img_subplot = plt.subplot(grid[0, 0]) # image
    dist_subplot = plt.subplot(grid[0, 1:]) # prob dist
    fig.add_subplot(img_subplot)
    fig.add_subplot(dist_subplot)
    bins = []
    bins.append(plt.subplot(grid[1, 0])) # bin1
    bins.append(plt.subplot(grid[1, 1])) # bin2
    bins.append(plt.subplot(grid[1, 2])) # bin3
    bins.append(plt.subplot(grid[1, 3])) # bin4
    bins.append(plt.subplot(grid[1, 4])) # bin5
    for bin_ in bins:
        fig.add_subplot(bin_)
    
    img_subplot.imshow(img.numpy().swapaxes(0,1).swapaxes(1,2)*0.5 + 0.5)
    sz = img.shape[-1]
    this_pred_bbox = out_bbox[0]*sz
    x1, y1, x2, y2 = this_pred_bbox
    rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                             linewidth=1,
                             edgecolor='r',
                             facecolor='none')
    img_subplot.add_patch(rect)

    # Plot p(y|x)
    dist_subplot.bar(np.arange(0, len(out_cls[0])),
                     torch.softmax(out_cls/softmax_temp, dim=1)[0])
    dist_subplot.set_xticklabels(cat_names, rotation='vertical')
    dist_subplot.set_xticks(np.arange(0, len(cat_names)))
    dist_subplot.set_xlabel("category")
    dist_subplot.set_ylabel("p(y|x)")

    # Plot attribute prediction.
    annos = dict()
    for j in range(1, 6):
        out_bin_subset = out_bin.clone()
        out_bin_subset[:, (attr_types != j)] = -1000.0
        #print( (out_bin_subset != -1000.0).sum() )
        this_topk_values, this_topk_indices = out_bin_subset.topk(top_k, 1, True, True)
        this_topk_values = torch.sigmoid(this_topk_values[0])
        this_topk_indices = this_topk_indices.numpy()[0]
        this_attr_names = attr_names[this_topk_indices]    
        annos[ attr_type_names[j-1] ] = (this_attr_names, this_topk_values)
    
    for i, key in enumerate(annos.keys()):
        names, probs = annos[key]
        probs = np.asarray([x.item() for x in probs])
        probs_norm = probs / sum(probs)
        labels = [ "%s\n(%f)" % (name,prob) for name,prob in zip(names, probs_norm) ]
        bins[i].pie(probs_norm, labels=labels)
        bins[i].set_title(key)

    fig.savefig(out_file)
    plt.close(fig)

model = FashionResnet(50, 1000, args.resnet_type)

from train import load_model
load_model(model, args.checkpoint, optimizer=None, devices=[])

if args.filenames is not None:
    # Process a comma-separated list of filenames.
    filenames = args.filenames.split(",")
    for filename in filenames:
        print("Processing %s..." % filename)
        out_file = "%s/%s" % (args.out_folder, filename)
        util.mkdir(os.path.dirname(out_file))
        plot_classification(net=model,
                            filename=filename,
                            out_file=out_file,
                            ensemble=True,
                            softmax_temp=args.softmax_temp,
                            top_k=args.top_k)
else:
    print("Processing %s..." % args.filename)
    util.mkdir(os.path.dirname(args.out_file))
    plot_classification(net=model,
                        filename=args.filename,
                        out_file=args.out_file,
                        ensemble=True,
                        softmax_temp=args.softmax_temp,
                        top_k=args.top_k)
