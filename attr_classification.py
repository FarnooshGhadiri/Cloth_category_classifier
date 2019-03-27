import os
import sys
import time
import logging
import numpy as np
import torch
from operator import add
import torch.nn as nn
import torch.optim as optim
from data.dataset_fashion import (DeepFashionDataset,
                                  get_list_attr_img,
                                  get_list_category_img)
from models.model import Fashion_model, save_model
from options.options import Options
from torch.utils.data import DataLoader
from util import util
from data.input import get_attr_name, get_Ctg_name, get_weight_attr_img
from random import sample
from tqdm import tqdm
import pickle

def forward_batch(model, criterion_softmax, criterion_binary, inputs, target_softmax,target_binary, opt, phase):
    if opt.cuda:
        inputs = inputs.cuda(opt.devices[0])

    if phase in ["Train"]:
        inputs_var = inputs
        # logging.info("Switch to Train Mode")
        model.train()
    elif phase in ["Validate", "Test"]:
        with torch.no_grad():
          inputs_var = inputs
          # logging.info("Switch to Test Mode")
          model.eval()

    # forward
    if opt.cuda:
        if len(opt.devices) > 1:
            output_softmax,output_binary = nn.parallel.data_parallel(model, inputs_var, opt.devices)
        else:
            if phase in ["Train"]:
                output_softmax, output_binary = model(inputs_var)
            elif phase in ["Validate", "Test"]:
                with torch.no_grad():
                  output_softmax, output_binary = model(inputs_var)
    else:
        output_softmax, output_binary = model(inputs_var)
        inputs = torch.cat((inputs,inputs), dim=0)
        #print(inputs_var.shape, output_softmax.min(),output_binary.min())

    # Calculate loss for sigmoid
    if opt.cuda:
        target_binary = target_binary.cuda(opt.devices[0])
    bin_loss = criterion_binary(output_binary, target_binary)

    # calculate loss for softmax

    if opt.cuda:
         target_softmax = target_softmax.cuda(opt.devices[0])

    cls_loss = criterion_softmax(output_softmax, target_softmax)
    #print(output_binary.min())
    return output_binary, output_softmax, bin_loss, cls_loss



def forward_dataset(model, criterion_softmax, criterion_binary, data_loader, opt):
    sum_batch = 0
    avg_accuracy = torch.zeros(len(opt.top_k))
    Sum_Num_correct_attr_type = torch.zeros(len(opt.top_k),6,dtype=torch.float)
    avg_acc_per_type = torch.zeros(len(opt.top_k),6,dtype=torch.float)
    Sum_Num_GT_attr_per_class = torch.zeros(len(opt.top_k),6,dtype=torch.float)
    Sum_Correct_pred_pre_attr = torch.zeros(len(opt.top_k),opt.numattri,dtype=torch.float)
    Sum_GT_attr_per_attr = torch.zeros(len(opt.top_k),opt.numattri,dtype=torch.float)
    avg_acc_per_attr = torch.zeros(len(opt.top_k),opt.numattri,dtype=torch.float)

    _, attr_type = get_attr_name(opt)
    avg_loss = [0,0]
    for i, data in enumerate(data_loader):
        #print(len(data_loader))
        if opt.mode == "Test":
            logging.info("test %s/%s image" % (i, len(data_loader)))

        sum_batch += 1
        inputs, target_softmax,target_binary = data

        output_binary, output_softmax, bin_loss, cls_loss = forward_batch(model, criterion_softmax, criterion_binary, inputs, target_softmax,target_binary, opt, "Validate")
        acc_softmax, Num_correct_attr_type, Num_GT_attr_per_class, Correct_pred_pre_attr, GT_attr_per_attr = calc_accuracy(output_binary, output_softmax, target_softmax,target_binary, opt.score_thres, opt.top_k,attr_type)

          # accumulate accuracy
        for k in range(len(opt.top_k)):
            avg_accuracy[k] = acc_softmax[k] + avg_accuracy[k]
            Sum_Num_correct_attr_type[k] = Sum_Num_correct_attr_type[k] + Num_correct_attr_type[k]
            Sum_Num_GT_attr_per_class[k] = Sum_Num_GT_attr_per_class[k] + Num_GT_attr_per_class[k]
            Sum_Correct_pred_pre_attr[k] = Sum_Correct_pred_pre_attr[k] + Correct_pred_pre_attr[k]
            Sum_GT_attr_per_attr[k] = Sum_GT_attr_per_attr[k] + GT_attr_per_attr[k]
        avg_loss = list( map(add, avg_loss, [bin_loss,cls_loss]))

    # average on batches
    avg_accuracy /=float(sum_batch)
    for k in range(len(opt.top_k)):
       avg_acc_per_type[k] = Sum_Num_correct_attr_type[k]/Sum_Num_GT_attr_per_class[k]
       avg_acc_per_attr[k] = Sum_Correct_pred_pre_attr[k]/Sum_GT_attr_per_attr[k]
    #avg_accuracy_bin /= float(sum_batch)
    avg_loss = map(lambda x: x/(sum_batch), avg_loss)
    return avg_accuracy, avg_acc_per_type,avg_acc_per_attr, avg_loss








def indices_to_binary(indices, length):
    tmp = torch.zeros(length).float()
    tmp[indices] = 1
    return tmp

def calc_accuracy(output_binary, output_softmax, target_softmax,target_binary, score_thres, top_k, attr_type_num):

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
                bin_acc_dict["bin_acc%i_top%i" % (j, k)] = np.mean(recalls_at_k[j])

        # Merge the two dictionaries together.
        cls_acc_dict.update(bin_acc_dict)

        return cls_acc_dict


def train(model, criterion_softmax, criterion_binary, train_set, val_set, opt):

    optimizer = optim.Adam(model.parameters(),lr=opt.lr)

    # record forward and backward times
    train_batch_num = len(train_set)
    total_batch_iter = 0
    logging.info("####################Train Model###################")
    _, attr_type = get_attr_name(opt)

    csv_file = "%s/results.txt" % opt.model_dir
    if not os.path.exists(csv_file):
        write_mode = 'w'
    else:
        write_mode = 'a'
    f_csv = open("%s/results.txt" % opt.model_dir, write_mode)
    
    for epoch in range(opt.epochs):
        epoch_start_t = time.time()
        logging.info('Begin of epoch %d' % (epoch))
        
        total_dict = {} # record per-minibatch metrics
        pbar = tqdm(total=len(train_set))
        
        for i, data in enumerate(train_set):
            inputs, target_softmax,target_binary = data
            
           # print(inputs.size())
           # print(target_binary.size())
            output_binary, output_softmax, bin_loss, cls_loss = forward_batch(
                model,
                criterion_softmax, criterion_binary,
                inputs,
                target_softmax, target_binary,
                opt, "Train")
            optimizer.zero_grad()
            loss = bin_loss + cls_loss
            loss_list = [bin_loss.item(), cls_loss.item()]
            loss.backward()
            optimizer.step()

            total_batch_iter += 1

            # display train loss and accuracy
            if epoch % opt.display_train_freq == 0:
                # accuracy

                acc_outputs = calc_accuracy(output_binary,
                                            output_softmax,
                                            target_softmax, target_binary,
                                            opt.score_thres,
                                            opt.top_k,
                                            attr_type)


                # save snapshot
            if total_batch_iter % opt.save_batch_iter_freq == 0:
               logging.info("saving the latest model (epoch %d, total_batch_iter %d)" % (epoch, total_batch_iter))
               save_model(model, opt, epoch)
               # TODO snapshot loss and accuracy

            pbar.update(1)
            pbar_dict = {'epoch': epoch+1, 'bin_loss': loss_list[0], 'cls_loss': loss_list[1]}
            pbar_dict.update(acc_outputs)
            pbar.set_postfix(pbar_dict)

            for key in pbar_dict:
                if key not in total_dict:
                    total_dict[key] = []
                total_dict[key].append(pbar_dict[key])

            # validate and display validate loss and accuracy
        pbar.close()

        if write_mode == 'w' and epoch == 0:
            csv_header = ",".join([key for key in total_dict])
            f_csv.write(csv_header + "\n")
            f_csv.flush()
        csv_values = ",".join([ str(np.mean(total_dict[key])) for key in total_dict ])
        f_csv.write(csv_values + "\n")
        f_csv.flush()

        '''
        if len(val_set) > 0:
           avg_val_accuracy, avg_acc_per_type, avg_acc_per_attr, avg_val_loss = validate(model, criterion_softmax, criterion_binary, val_set, opt)
           util.print_loss(avg_val_loss, "Validate", epoch, total_batch_iter)
           util.print_accuracy_soft(avg_val_accuracy, "Validate", epoch, total_batch_iter)
           util.print_accuracy_attr(avg_acc_per_type,avg_acc_per_attr, "Validate", epoch, total_batch_iter)
           #util.print_accuracy_bin(avg_val_accuracy_bin, "Validate", epoch, total_batch_iter)
               # if opt.display_id > 0:
                  #  webvis.plot_points(x_axis, val_loss, "Loss", "Validate")
                  # webvis.plot_points(x_axis, accuracy_list, "Accuracy", "Validate")
        '''


      #  logging.info('End of epoch %d / %d \t Time Taken: %d sec' %
                    # (epoch, opt.sum_epoch, time.time() - epoch_start_t))

        if epoch % opt.save_epoch_freq == 0:
            logging.info('saving the model at the end of epoch %d, iters %d' % (epoch + 1, total_batch_iter))
            save_model(model, opt, epoch + 1)

            # adjust learning rate
        #lr = optimizer.param_groups[0]['lr']
        #logging.info('learning rate = %.7f epoch = %d' % (lr, epoch))
    logging.info("--------Optimization Done--------")


def validate(model, criterion_softmax, criterion_binary, val_set, opt):
    return forward_dataset(model, criterion_softmax, criterion_binary, val_set, opt)


def test(model, criterion_softmax, criterion_binary, test_set, opt):
    logging.info("####################Test Model###################")
    avg_test_accuracy, avg_acc_per_type, avg_acc_per_attr, avg_test_loss = forward_dataset(model, criterion_softmax,
                                                                                         criterion_binary, test_set,
                                                                                         opt)
    util.print_loss(avg_test_loss, "Test", 0, 0)
    util.print_accuracy_soft(avg_test_accuracy, "Test", 0, 0)
    util.print_accuracy_attr(avg_acc_per_type, avg_acc_per_attr, "Test", 0, 0)
    logging.info("data_dir:   " + opt.data_dir + "/TestSet/")
    logging.info("score_thres:" + str(opt.score_thres))

    for k in range(0, avg_acc_per_type.size(0)):
        logging.info("----Accuracy of Top%d: %f" % (opt.top_k[k], avg_acc_per_type[k, 0]))

    logging.info("----Average test accuracy %f" % (avg_test_accuracy))
    logging.info("#################Finished Testing################")


def main():


    print("parse opt...")
    
    # parse options
    op = Options()
    opt = op.parse()

    # initialize train or test working dir
    trainer_dir = "trainer_" + opt.name
    opt.model_dir = os.path.join("results", trainer_dir, opt.name)
    logging.info("Model directory: %s" % opt.model_dir)
    #opt.data_dir = os.path.join(opt.data_dir, trainer_dir, "Data")
    opt.test_dir = os.path.join(opt.data_dir, trainer_dir, "Test")
    logging.info("Test directory: %s" % opt.test_dir)

    # why need to make the data dir??
    #if not os.path.exists(opt.data_dir):
    #    os.makedirs(opt.data_dir)
    if opt.mode == "Train":
        if not os.path.exists(opt.model_dir):
            os.makedirs(opt.model_dir)
        log_dir = opt.model_dir
        log_path = log_dir + "/train.log"
    if opt.mode == "Test":
        if not os.path.exists(opt.test_dir):
            os.makedirs(opt.test_dir)
        log_dir = opt.test_dir
        log_path = log_dir + "/test.log"

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

    filenames, attrs = get_list_attr_img(opt.data_dir)
    categories = get_list_category_img(opt.data_dir)
    
    indices = list(range(len(filenames)))
    rnd_state = np.random.RandomState(0)
    rnd_state.shuffle(indices)
    train_idx = indices[0:int(0.9*len(indices))]
    valid_idx = indices[int(0.9*len(indices)):int(0.95*len(indices))]
    test_idx = indices[int(0.95*len(indices))::]

    # Define datasets.
    ds_train = DeepFashionDataset(root=opt.data_dir,
                                  filenames=filenames,
                                  indices=train_idx,
                                  attrs=attrs,
                                  categories=categories,
                                  img_size=opt.img_size,
                                  crop_size=opt.crop_size)
    ds_valid = DeepFashionDataset(root=opt.data_dir,
                                  filenames=filenames,
                                  indices=valid_idx,
                                  attrs=attrs,
                                  categories=categories,
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
                              num_workers=1)
    '''
    loader_test = DataLoader(ds_train,
                             shuffle=False,
                             batch_size=opt.batch_size,
                             num_workers=1)
    '''
    

    num_classes = [opt.numctg, opt.numattri] #temporary lets put the number of class []
    opt.class_num = len(num_classes)

    # load model
    model = Fashion_model(opt, num_classes)
    logging.info(model)

    # imagenet pretrain model
    if opt.pretrain:
        logging.info("use pretrained model")
    # load exsiting model
    if opt.checkpoint_name != "":
        if os.path.exists(opt.checkpoint_name):
            logging.info("load pretrained model from " + opt.checkpoint_name)
            model.load_state_dict(torch.load(opt.checkpoint_name))
        elif os.path.exists(opt.model_dir):
            checkpoint_name = opt.model_dir + "/" + opt.checkpoint_name
            model.load_state_dict(torch.load(checkpoint_name))
            logging.info("load pretrained model from " + checkpoint_name)
        else:
            opt.checkpoint_name = ""
            logging.warning("WARNING: unknown pretrained model, skip it.")


    #Weight_attribute = get_weight_attr_img(opt)
   # print(len(Weight_attribute))
    # define loss function
    criterion_softmax = nn.CrossEntropyLoss() #weight=opt.loss_weight
    criterion_binary = torch.nn.BCELoss()


    # use cuda
    if opt.cuda:
        model = model.cuda(opt.devices[0])
        criterion_softmax = criterion_softmax.cuda(opt.devices[0])
        criterion_binary = criterion_binary.cuda(opt.devices[0])
        #cudnn.benchmark = True

    # Train model
    if opt.mode == "Train":
        train(model, criterion_softmax,criterion_binary, loader_train, loader_valid, opt)
    # Test model
    elif opt.mode == "Test":
        test(model, criterion_softmax,criterion_binary, loader_valid, opt)


if __name__ == "__main__":
    main()
