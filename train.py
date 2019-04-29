import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import patches
import os
import sys
import time
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from data.dataset_fashion import (DeepFashionDataset,
                                  get_list_attr_img,
                                  get_list_category_img,
                                  get_weight_attr_img,
                                  get_bboxes)
from models.model import FashionResnet
from options.options import Options
from torch.utils.data import DataLoader
from util import util
from data.input import get_attr_name, get_ctg_name
from tqdm import tqdm
from collections import OrderedDict

amp_imported = True
try:
    from apex import amp
except ImportError:
    logging.warning("""Tried to import apex but failed.
    (Don't worry about this error, unless you have --fp16
    enabled.)
    """)
    amp_imported = False

def forward_batch(model, criterion_softmax, criterion_binary, inputs, target_softmax, target_binary, target_bbox, opt, phase):
    """Perform a forward pass on a batch.

    :param model:
    :param criterion_softmax: loss function for category classification
    :param criterion_binary: loss function for attribute classification
    :param inputs: inputs from DataLoader
    :param target_softmax: class label for category classification
    :param target_binary: attribute label for attribute classificatiion
    :param target_bbox: label for bounding box regression
    :param opt: options dictionary
    :param phase: either 'train', 'valid', or 'test'
    :returns: a tuple in the form: (attribute output, classification output,
      bbox output, attribute loss, classification loss, and bbox loss)
    """
    if opt.cuda:
        inputs = inputs.cuda(opt.devices[0])
        target_binary = target_binary.cuda(opt.devices[0])
        target_softmax = target_softmax.cuda(opt.devices[0])
        target_bbox = target_bbox.cuda(opt.devices[0])
        
    if phase in ["train"]:
        model.train()
    elif phase in ["valid", "test"]:
        model.eval()
    else:
        raise Exception("phase must be either train, valid, or test!")

    if phase in ["train"]:
        if len(opt.devices) > 1:
            output_softmax, output_binary, output_bbox = nn.parallel.data_parallel(model, inputs, opt.devices)
        else:
            output_softmax, output_binary, output_bbox = model(inputs)
    elif phase in ["valid", "test"]:
        with torch.no_grad():
            output_softmax, output_binary, output_bbox = model(inputs)

    # Calculate loss for attributes.
    bin_loss = criterion_binary(output_binary, target_binary)
    if opt.reduce_sum:
        bin_loss = bin_loss.sum(dim=1).mean()
    else:
        bin_loss = bin_loss.mean()
    # Calculate loss for classification.
    cls_loss = criterion_softmax(output_softmax, target_softmax)
    # Calculate loss for bounding boxes.
    bbox_loss = torch.mean(torch.abs(output_bbox-target_bbox))
    return output_binary, output_softmax, output_bbox, bin_loss, cls_loss, bbox_loss

def forward_dataset(model, criterion_softmax, criterion_binary, loader, opt):
    """Perform a forward pass through a dataset.

    :param model: 
    :param criterion_softmax: loss function for category classification
    :param criterion_binary:  loss function for attribute classification
    :param loader: data loader to perform the forward pass through
    :param opt: option dictionary
    :returns: a dictionary of relevant statistics
    """
    
    _, attr_type = get_attr_name(opt.data_dir)
    with torch.no_grad():
        stats = OrderedDict()
        stats['bbox_loss'] = []
        pbar = tqdm(total=len(loader))
        for i, data in enumerate(loader):
            filepaths, inputs, target_softmax, target_binary, target_bbox = data
            output_binary, output_softmax, output_bbox, bin_loss, cls_loss, bbox_loss = forward_batch(
                model,
                criterion_softmax, criterion_binary,
                inputs,
                target_softmax, target_binary, target_bbox,
                opt,
                'test')
            acc_outputs = calc_accuracy(output_binary,
                                        output_softmax,
                                        target_softmax, target_binary,
                                        opt.top_k,
                                        attr_type)
            pbar.update(1)
            for key in acc_outputs:
                if key not in stats:
                    stats[key] = []
                stats[key].append(acc_outputs[key])
            stats['bbox_loss'].append(bbox_loss.item())
        pbar.close()
    return stats


def save_model(model, optimizer, model_dir, epoch):
    """Save model and its optimizer state.

    :param model: model whose state is to be saved
    :param optimizer: optimizer whose state is to be saved
    :param model_dir: where to save the model
    :param epoch: epoch # to save
    :returns: 
    """
    checkpoint_name = "%s/epoch_%s.pth" % (model_dir, epoch)
    dd = {'model': model.state_dict(),
          'optim': optimizer.state_dict(),
          'epoch': epoch}
    torch.save(dd, checkpoint_name)

def load_model(model, checkpoint, optimizer=None, devices=[]):
    """Load model and its optimizer state.

    :param model: model whose weights are to be loaded
    :param optimizer: optimizer
    :param checkpoint: path to the checkpoint file
    :param devices:
    :returns:
    """
    
    dd = torch.load(checkpoint)
    model.load_state_dict(dd['model'])
    if 'optim' in dd and optimizer is not None:
        optimizer.load_state_dict(dd['optim'])
        for p in optimizer.state.keys():
            param_state = optimizer.state[p]
            for key in param_state:
                if hasattr(param_state[key], 'cuda'):
                    param_state[key] = param_state[key].cuda(devices[0])
    return dd['epoch']

def calc_accuracy(output_binary, output_softmax, target_softmax, target_binary, top_k, attr_type_num):
    """Compute accuracy and recall metrics.
    TODO: this is quite a hefty method which may need a bit of refactoring

    :param output_binary: output from attribute classification
    :param output_softmax: output from category classfication
    :param target_softmax: ground truth for category classification
    :param target_binary: ground truth for attribute classification
    :param top_k: tuple of numbers (k1,k2,...) such that we compute
      metrics for top-k1, top-k2, etc.
    :param attr_type_num: a vector v of integers (same length as the #
      of attributes in the dataset) where v[i] \in {1,..5} denotes the
      'attribute category' of that particular attribute.
    :returns: 
    """

    with torch.no_grad():
    
        max_k = max(top_k)
        batch_size = target_softmax.size(0)
        _, pred_cls = output_softmax.data.cpu().topk(max_k, 1, True, True)
        pred_cls = pred_cls.t()
        correct = pred_cls.eq(target_softmax.cpu().view(1, -1).expand_as(pred_cls))
        cls_acc_dict = {}
        for k in top_k:
            correct_k = correct[:k].view(-1).float().sum(0).item()
            cls_acc_dict["cls_acc_top%i" % k] = correct_k / batch_size

        bin_acc_dict = {}
        for k in top_k:
            # Get the top-k indices for the attribute prediction.
            _, pred_bin = output_binary.data.cpu().topk(k, 1, True, True)

            # Ok, we want to compute the recall@k for each of the attribute
            # categories.
            recalls_at_k = {j:[] for j in range(1, 6)}
            for j in range(1, 6):
                mask_j = (attr_type_num == j).float()
                mask_not_j = (attr_type_num != j).long()
                # Get the binary predictions, then make -ve the attribute indices
                # which are NOT attribute j. That way, when we consider top-k for
                # predictions, it is only amongst the indices coding for category j.
                preds_for_j = output_binary.data.cpu()
                preds_for_j[:, mask_not_j] = -1.
                _, preds_topk = preds_for_j.topk(k, 1, True, True)
                # TODO: batchify this??
                for b in range(batch_size):
                    # Extract the ground truth indices for attribute category j.
                    this_gt_indices = np.where( (target_binary[b, :]*mask_j).numpy() )[0]
                    # Extract the top-k indices for attribute category j.
                    this_pred_indices = preds_topk[b, :].numpy()
                    if len(this_gt_indices) > 0:
                        # If there are ground truth indices associated with category j...
                        n_correct = len(set(this_gt_indices).intersection(set(this_pred_indices)))
                        denominator = min(k, len(this_gt_indices))
                        recalls_at_k[j].append(n_correct / denominator)
            for j in range(1, 6):
                # to avoid nan'ing
                if len(recalls_at_k[j]) > 0:
                    bin_acc_dict["bin_acc%i_top%i" % (j, k)] = np.mean(recalls_at_k[j])
        # Merge the two dictionaries together.
        cls_acc_dict.update(bin_acc_dict)
        return cls_acc_dict

def bbox_on_image(img_batch, gt_bbox_batch, pred_bbox_batch, out_file):
    n_images = img_batch.size(0)
    fig, axes = plt.subplots(1, n_images)
    fig.set_figheight(10)
    fig.set_figwidth(10)
    img_batch = img_batch.numpy()*0.5 + 0.5
    sz = img_batch.shape[3]
    for i in range(n_images):
        if n_images == 1:
            this_ax = axes
        else:
            this_ax = axes[i]
        this_ax.imshow(img_batch[i].swapaxes(0,1).swapaxes(1,2))
        # (x1, y1, x2, y2)
        this_gt_bbox = gt_bbox_batch[i]*sz
        x1, y1, x2, y2 = this_gt_bbox
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                 linewidth=1,
                                 edgecolor='b',
                                 facecolor='none')
        this_ax.add_patch(rect)
        this_pred_bbox = pred_bbox_batch[i]*sz
        x1, y1, x2, y2 = this_pred_bbox
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                 linewidth=1,
                                 edgecolor='r',
                                 facecolor='none')
        this_ax.add_patch(rect)
    plt.savefig(out_file)
    plt.close(fig)

    
def train(model, optimizer, criterion_softmax, criterion_binary, train_loader, val_loader, opt, epoch=0):

    # record forward and backward times
    logging.info("####################Train Model###################")
    _, attr_type = get_attr_name(opt.data_dir)

    csv_file = "%s/results.txt" % opt.model_dir
    if not os.path.exists(csv_file):
        write_mode = 'w'
    else:
        write_mode = 'a'
    f_csv = open("%s/results.txt" % opt.model_dir, write_mode)

    last_epoch = epoch
    
    for epoch in range(last_epoch, opt.epochs):
        start_time = time.time()
        
        total_dict = {} # record per-minibatch metrics

        for loader_tuple in [('train', train_loader), ('valid', val_loader)]:
            loader_name, loader = loader_tuple
            
            pbar = tqdm(total=len(loader))
            for i, data in enumerate(loader):
                if loader_name == 'train':
                    optimizer.zero_grad()
                filepaths, inputs, target_softmax, target_binary, target_bbox = data
                output_binary, output_softmax, output_bbox, bin_loss, cls_loss, bbox_loss = forward_batch(
                    model,
                    criterion_softmax, criterion_binary,
                    inputs,
                    target_softmax, target_binary, target_bbox,
                    opt,
                    loader_name)
                loss = bin_loss + cls_loss + opt.beta*bbox_loss
                
                if loader_name == 'train':
                    if opt.fp16:
                        with amp.scale_loss(loss, optimizer) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        loss.backward()
                    optimizer.step()

                acc_outputs = calc_accuracy(output_binary,
                                            output_softmax,
                                            target_softmax, target_binary,
                                            opt.top_k,
                                            attr_type)

                pbar.update(1)
                pbar_dict = {'epoch': epoch+1,
                             '%s_bin_loss' % loader_name: bin_loss.item(),
                             '%s_cls_loss' % loader_name: cls_loss.item(),
                             '%s_bbox_loss' % loader_name: bbox_loss.item()}
                for key in acc_outputs:
                    pbar_dict["%s_%s" % (loader_name, key)] = acc_outputs[key]
                pbar.set_postfix(pbar_dict)

                for key in pbar_dict:
                    if key not in total_dict:
                        total_dict[key] = []
                    total_dict[key].append(pbar_dict[key])

                if i == 0:
                    bbox_on_image(inputs, target_bbox, output_bbox.detach(),
                                  "%s/bbox_%i.png" % (opt.model_dir, epoch+1))
                    

        pbar.close()

        total_dict['time'] = time.time() - start_time

        if write_mode == 'w' and epoch == 0:
            csv_header = ",".join([key for key in total_dict])
            f_csv.write(csv_header + "\n")
            f_csv.flush()
        csv_values_comma = ",".join([ str(np.mean(total_dict[key])) for key in total_dict ])
        csv_values_tab = "\t".join([ str(np.mean(total_dict[key])) for key in total_dict ])
        print(csv_values_tab)
        f_csv.write(csv_values_comma + "\n")
        f_csv.flush()

        if (epoch+1) % opt.save_every == 0:
            save_model(model, optimizer, opt.model_dir, epoch + 1)

    logging.info("--------Optimization Done--------")

def validate(model, criterion_softmax, criterion_binary, val_set, opt):
    return forward_dataset(model, criterion_softmax, criterion_binary, val_set, opt)

class HingeLoss(nn.Module):
    def __init__(self):
        super(HingeLoss, self).__init__()
    def forward(self, output, target):
        target_ctr = (target - 0.5) / 0.5
        return nn.ReLU()(1. - target_ctr*output)
    
def main():

    print("parse opt...")
    
    # parse options
    op = Options()
    opt = op.parse()

    # initialize train or test working dir
    opt.model_dir = os.path.join("results", opt.name)
    logging.info("Model directory: %s" % opt.model_dir)

    if not os.path.exists(opt.model_dir):
        os.makedirs(opt.model_dir)
    log_dir = opt.model_dir
    log_path = log_dir + "/train.log"

    # save options to disk
    util.opt2file(opt, log_dir + "/opt.txt")

    # log setting
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(log_format)
    fh = logging.FileHandler(log_path, 'a')
    fh.setFormatter(formatter)
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logging.getLogger().addHandler(fh)
    logging.getLogger().addHandler(ch)
    log_level = logging.INFO
    logging.getLogger().setLevel(log_level)

    '''
    pkl_file = "%s/metadata.pkl" % opt.data_dir
    if not os.path.exists(pkl_file):
        # If metadata file does not exist, manually create it
        # from the txt files and save a pkl.
        filenames, attrs = get_list_attr_img(opt.data_dir)
        categories = get_list_category_img(opt.data_dir)
        with open(pkl_file, "wb") as f:
            pickle.dump({'filenames': filenames,
                         'attrs': attrs,
                         'categories': categories}, f)
    else:
        logging.info("Found %s..." % pkl_file)
        with open(pkl_file, "rb") as f:
            dat = pickle.load(f)
            filenames = dat['filenames']
            attrs = dat['attrs']
            categories = dat['categories']
    '''

    attrs = get_list_attr_img(opt.data_dir)
    categories = get_list_category_img(opt.data_dir)
    bboxes = get_bboxes(opt.data_dir)
    
    indices = list(range(len(attrs.keys())))
    rnd_state = np.random.RandomState(0)
    rnd_state.shuffle(indices)
    train_idx = indices[0:int(0.9*len(indices))]
    valid_idx = indices[int(0.9*len(indices)):int(0.95*len(indices))]
    test_idx = indices[int(0.95*len(indices))::]

    # Define datasets.
    ds_train = DeepFashionDataset(root=opt.data_dir,
                                  indices=train_idx,
                                  attrs=attrs,
                                  categories=categories,
                                  bboxes=bboxes,
                                  data_aug=opt.data_aug,
                                  img_size=opt.img_size,
                                  crop_size=opt.crop_size)
    ds_valid = DeepFashionDataset(root=opt.data_dir,
                                  indices=valid_idx,
                                  attrs=attrs,
                                  categories=categories,
                                  bboxes=bboxes,
                                  data_aug=opt.data_aug,
                                  img_size=opt.img_size,
                                  crop_size=opt.crop_size)
    '''
    ds_test = DeepFashionDataset(root=opt.data_dir,
                                 indices=test_idx,
                                 img_size=opt.img_size,
                                 crop_size=opt.crop_size)
    '''
    # Define data loaders.
    loader_train = DataLoader(ds_train,
                              shuffle=True,
                              batch_size=opt.batch_size,
                              num_workers=opt.num_workers)
    loader_valid = DataLoader(ds_valid,
                              shuffle=False,
                              batch_size=opt.batch_size,
                              num_workers=opt.num_workers)
    '''
    loader_test = DataLoader(ds_train,
                             shuffle=False,
                             batch_size=opt.batch_size,
                             num_workers=1)
    '''

    # load model
    model = FashionResnet(50, 1000, opt.resnet_type)
    logging.info(model)

    if opt.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=opt.lr, eps=opt.eps)
    else:
        optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9)
    
    # load exsiting model
    last_epoch = 0
    if opt.resume is not None:
        if opt.resume == 'auto':
            import glob
            # List all the pkl files.
            files = glob.glob("%s/*.pth" % opt.model_dir)
            # Make them absolute paths.
            files = [os.path.abspath(key) for key in files]
            if len(files) > 0:
                # Get creation time and use that.
                latest_chkpt = max(files, key=os.path.getctime)
                logging.info("Auto-resume mode found latest checkpoint: %s" % latest_chkpt)
                last_epoch = load_model(model, latest_chkpt, optimizer, devices=opt.devices)
        else:
            logging.info("Loading checkpoint: %s" % opt.resume)
            last_epoch = load_model(model, opt.resume, optimizer, devices=opt.devices)

    #Weight_attribute = get_weight_attr_img(opt)
   # print(len(Weight_attribute))
    # define loss function
    criterion_softmax = nn.CrossEntropyLoss() #weight=opt.loss_weight
    if opt.loss == 'bce':
        if opt.pos_weights:
            logging.info("Using pos_weights...")
            pos_weights = (1-attrs).sum(dim=0) / attrs.sum(dim=0)
            # Scale pos_weights such that its maximum value will be == pos_weights_scale.
            # This is in case pos_weights has too big of a range.
            pos_weights = pos_weights / ( pos_weights.max() / opt.pos_weights_scale )
            criterion_binary = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weights,
                                                          reduction='none')
        else:
            criterion_binary = torch.nn.BCEWithLogitsLoss(reduction='none')
    else:
        if opt.pos_weights:
            raise Exception("`pos_weights` only works with BCE loss!")
        criterion_binary = HingeLoss()

    # use cuda
    if opt.cuda:
        model = model.cuda(opt.devices[0])
        criterion_softmax = criterion_softmax.cuda(opt.devices[0])
        criterion_binary = criterion_binary.cuda(opt.devices[0])

    # float16
    if opt.fp16:
        if not amp_imported:
            raise Exception(
                """Was not able to import apex library. This is
                required for float16 mode."""
            )
        model, optimizer = amp.initialize(
            model,
            optimizer,
            enabled=True,
            opt_level='O1'
        )
       
    # Train model
    if opt.mode == "train":
        logging.info("Running in train mode")
        train(model=model,
              optimizer=optimizer,
              criterion_softmax=criterion_softmax,
              criterion_binary=criterion_binary,
              train_loader=loader_train,
              val_loader=loader_valid,
              opt=opt,
              epoch=last_epoch)
    # Test model
    elif opt.mode == "validate":
        logging.info("Running in validate mode")
        accs = forward_dataset(model, criterion_softmax, criterion_binary, loader_valid, opt)
        for key in accs:
            print("%s --> %.4f +/- %.4f" % (key, np.mean(accs[key]), np.std(accs[key])))
    elif opt.mode == "test":
        logging.info("Running in test mode")
        ds_test = DeepFashionDataset(root=opt.data_dir,
                                     indices=test_idx,
                                     attrs=attrs,
                                     categories=categories,
                                     bboxes=bboxes,
                                     img_size=opt.img_size,
                                     crop_size=opt.crop_size)
        loader_test = DataLoader(ds_test,
                                 shuffle=False,
                                 batch_size=opt.batch_size,
                                 num_workers=opt.num_workers)
        accs = forward_dataset(model, criterion_softmax, criterion_binary, loader_test, opt)
        for key in accs:
            print("%s --> %.4f +/- %.4f" % (key, np.mean(accs[key]), np.std(accs[key])))


if __name__ == "__main__":
    main()
