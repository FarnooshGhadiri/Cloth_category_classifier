from __future__ import print_function

import os
import sys
import time
import copy
import random
import logging
import numpy as np
import torch
from operator import add
print("Pytorch Version: ", torch.__version__)
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from collections import OrderedDict, defaultdict
from data.dataset_fashion import DeepFashionDataset
from models.model import Fashion_model, save_model
from options.options import Options
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from util import util
from data.input import get_attr_name, get_Ctg_name,get_weight_attr_img
from tqdm import tqdm

#approach like this
#  https://www.ritchievink.com/blog/2018/04/12/transfer-learning-with-pytorch-assessing-road-safety-with-computer-vision/

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
    Sum_Num_correct_attr_type = torch.zeros(6,dtype=torch.float)
    Sum_Num_GT_attr_per_class = torch.zeros(6,dtype=torch.float)
    Sum_Correct_pred_pre_attr = torch.zeros(opt.numattri,dtype=torch.float)
    Sum_GT_attr_per_attr = torch.zeros(opt.numattri,dtype=torch.float)

    avg_loss = [0,0]
    for i, data in enumerate(data_loader):
        #print(len(data_loader))
        if opt.mode == "Test":
            logging.info("test %s/%s image" % (i, len(data_loader)))

        sum_batch += 1
        inputs, target_softmax,target_binary = data
        output_binary, output_softmax, bin_loss, cls_loss = forward_batch(model, criterion_softmax, criterion_binary, inputs, target_softmax,target_binary, opt, "Validate")

        acc_softmax, Num_correct_attr_type, Num_GT_attr_per_class, Correct_pred_pre_attr, GT_attr_per_attr = calc_accuracy(output_binary, output_softmax, target_softmax,target_binary, opt.score_thres, opt.top_k,opt)
        # accumulate accuracy
        for k in range(len(opt.top_k)):
            avg_accuracy[k] = acc_softmax[k] + avg_accuracy[k]

        Sum_Num_correct_attr_type = Sum_Num_correct_attr_type + Num_correct_attr_type
        Sum_Num_GT_attr_per_class = Sum_Num_GT_attr_per_class + Num_GT_attr_per_class
        Sum_Correct_pred_pre_attr = Sum_Correct_pred_pre_attr + Correct_pred_pre_attr
        Sum_GT_attr_per_attr = Sum_GT_attr_per_attr + GT_attr_per_attr

        avg_loss = list( map(add, avg_loss, [bin_loss,cls_loss]))

    # average on batches
    avg_accuracy /=float(sum_batch)
    avg_acc_per_type = Sum_Num_correct_attr_type/Sum_Num_GT_attr_per_class
    avg_acc_per_attr = Sum_Correct_pred_pre_attr/Sum_GT_attr_per_attr
    #avg_accuracy_bin /= float(sum_batch)
    avg_loss = map(lambda x: x/(sum_batch), avg_loss)
    return avg_accuracy, avg_acc_per_type,avg_acc_per_attr, avg_loss

def calc_accuracy(output_binary, output_softmax, target_softmax,target_binary, score_thres, top_k,opt):
    max_k = max(top_k)
    batch_size = target_softmax.size(0)
    _, pred = output_softmax.data.cpu().topk(max_k, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target_softmax.cpu().view(1, -1).expand_as(pred))
    acc_soft = []
    for k in top_k:
        correct_k = correct[:k].view(-1).float().sum(0)
        acc_soft.append(correct_k.mul_(100.0 / batch_size))

    #Calculating accuracy for attribute
    _, attr_type = get_attr_name(opt)
    # print(attr_type)
    Num_GT_attr_per_class = torch.zeros(6,dtype=torch.float) # all attr, Texture, fabric, Shape, part, style
    correct_attr_type = torch.zeros(6,dtype=torch.float) # all attr, Texture, fabric, Shape, part, style
    Sum_all_corr_attr = torch.zeros(opt.numattri)
    #pred = (output_binary.cpu()>=0.50
    _, pred2 = output_binary.data.cpu().topk(3, 1, True, True)
    tmp_output = torch.zeros(output_binary[0].size())
    for i in range(0, len(pred2)):
        tmp_output[pred[i]] = 1
        if i == 0:
            Topk_binary = tmp_output
        else:
            Topk_binary = torch.cat((tmp_output, Topk_binary), 0)

    Correct_pred_bin = Topk_binary.float() * target_binary
    Correct_pred_per_class = attr_type.repeat(batch_size, 1) * Correct_pred_bin
    correct_attr_type[0] = Correct_pred_bin.sum()
    correct_attr_type[1] = (Correct_pred_per_class == 1).sum()
    correct_attr_type[2] = (Correct_pred_per_class == 2).sum()
    correct_attr_type[3] = (Correct_pred_per_class == 3).sum()
    correct_attr_type[4] = (Correct_pred_per_class == 4).sum()
    correct_attr_type[5] = (Correct_pred_per_class == 5).sum()
     #Numeber GT per class
    Num_GT_attr = target_binary * attr_type.repeat(batch_size, 1)
    Num_GT_attr_per_class[0] = target_binary.sum()
    Num_GT_attr_per_class[1] = (Num_GT_attr == 1).sum()
    Num_GT_attr_per_class[2] = (Num_GT_attr == 2).sum()
    Num_GT_attr_per_class[3] = (Num_GT_attr == 3).sum()
    Num_GT_attr_per_class[4] = (Num_GT_attr == 4).sum()
    Num_GT_attr_per_class[5] = (Num_GT_attr == 5).sum()


    return acc_soft, correct_attr_type, Num_GT_attr_per_class, Correct_pred_bin.sum(0), target_binary.sum(0)


def train(model, criterion_softmax, criterion_binary, train_set, val_set, opt):

    optimizer = optim.Adam(model.parameters(),lr=opt.lr)

    # record forward and backward times
    train_batch_num = len(train_set)
    total_batch_iter = 0
    logging.info("####################Train Model###################")
    for epoch in range(opt.sum_epoch):
        epoch_start_t = time.time()
        epoch_batch_iter = 0
        logging.info('Begin of epoch %d' % (epoch))
        pbar = tqdm(total=len(train_set))
        for i, data in enumerate(train_set):
            inputs, target_softmax, target_binary = data
            #print(inputs.size())
           # print(target_binary.size())
            output_binary, output_softmax, bin_loss, cls_loss = forward_batch(
                model,
                criterion_softmax, criterion_binary,
                inputs,
                target_softmax, target_binary,
                opt,
                "Train")
            
            optimizer.zero_grad()
            loss = bin_loss + cls_loss
            loss_list = [bin_loss.item(), cls_loss.item()]
            loss.backward()
            optimizer.step()

           # webvis.reset()
            epoch_batch_iter += 1
            total_batch_iter += 1

            # display train loss and accuracy
            if total_batch_iter % opt.display_train_freq == 0:
                # accuracy

                batch_accuracy_soft,Num_correct_attr_type, Num_GT_attr_per_class, Correct_pred_pre_attr, GT_attr_per_attr = calc_accuracy(output_binary, output_softmax, target_softmax,target_binary, opt.score_thres, opt.top_k,opt)
                batch_accuracy_bin=Num_correct_attr_type/Num_GT_attr_per_class
                util.print_loss(loss_list, "Train", epoch, total_batch_iter)
                util.print_accuracy_soft(batch_accuracy_soft, "Train", epoch, total_batch_iter)
                util.print_accuracy_attr(batch_accuracy_bin, [], "Train", epoch, total_batch_iter)
                #util.print_accuracy_bin(batch_accuracy_bin, "Train", epoch, total_batch_iter)

                # save snapshot
            if total_batch_iter % opt.save_batch_iter_freq == 0:
               logging.info("saving the latest model (epoch %d, total_batch_iter %d)" % (epoch, total_batch_iter))
               save_model(model, opt, epoch)
               # TODO snapshot loss and accuracy

            # validate and display validate loss and accuracy

            pbar.update(1)
        if len(val_set) > 0:
           avg_val_accuracy, avg_acc_per_type, avg_acc_per_attr, avg_val_loss = validate(model, criterion_softmax, criterion_binary, val_set, opt)
           util.print_loss(avg_val_loss, "Validate", epoch, total_batch_iter)
           util.print_accuracy_soft(avg_val_accuracy, "Validate", epoch, total_batch_iter)
           util.print_accuracy_attr(avg_acc_per_type,avg_acc_per_attr, "Validate", epoch, total_batch_iter)
           #util.print_accuracy_bin(avg_val_accuracy_bin, "Validate", epoch, total_batch_iter)
               # if opt.display_id > 0:
                  #  webvis.plot_points(x_axis, val_loss, "Loss", "Validate")
                  # webvis.plot_points(x_axis, accuracy_list, "Accuracy", "Validate")
        pbar.close()


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
    test_accuracy, test_accuracy_type, test_acc_per_attr, test_loss = forward_dataset(model, criterion_softmax, criterion_binary, test_set, opt)
    logging.info("data_dir:   " + opt.data_dir + "/TestSet/")
    logging.info("score_thres:" + str(opt.score_thres))
    for i in range(len(test_accuracy_type)):
        logging.info("----Attribute %f" % (test_accuracy_type[i]))

    for i in range(len(test_acc_per_attr)):
        logging.info("----Attribute %f" % (test_acc_per_attr[i]))

    for index, item in enumerate(test_accuracy):
            logging.info("----Accuracy of Top%d: %f" % (opt.top_k[index], item))

    for index, item in enumerate(test_loss):
          logging.info("----Loss of ctg and attrs%d: %f" % (index, item))
    logging.info("#################Finished Testing################")


def main():

    print("parse opt...")
    
    # parse options
    op = Options()
    opt = op.parse()

    # initialize train or test working dir
    trainer_dir = "trainer_" + opt.name
    opt.model_dir = os.path.join("results", trainer_dir, "Train_res18_512")
    #opt.data_dir = os.path.join(opt.data_dir, trainer_dir, "Data")
    opt.test_dir = os.path.join(opt.data_dir, trainer_dir, "Test")

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

    print("opt2file...")

    # save options to disk
    util.opt2file(opt, log_dir + "/opt.txt")

    print("log setting...")

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

    print("load dataset...")

    # load train or test data
    ds = DeepFashionDataset(opt)
    num_data = len(ds)
    print(num_data)
    indices = list(range(num_data))
    if os.path.isfile("train_idx_%s%s" % (opt.Try,'.txt')):
        test_idx = np.loadtxt(open("Test_idx_%s%s" % (opt.Try, '.txt'), 'rb'))
        train_idx = np.loadtxt(open("train_idx_%s%s" % (opt.Try, '.txt'), 'rb'))
        validation_idx = np.loadtxt(open("validation_idx_%s%s" % (opt.Try, '.txt'), 'rb'))
    else:
        split = int((opt.ratio[0]) * num_data)
        train_idx = np.random.choice(indices, size=split, replace=False)
       # print(len(train_idx))
        validation_Test_idx = list(set(indices) - set(train_idx))
        #save train_idx
        np.savetxt("train_idx_%s%s" % (opt.Try,'.txt'), train_idx, fmt='%i', delimiter=",")
        split = int(round(opt.ratio[1] * len(validation_Test_idx)))
        validation_idx = np.random.choice(validation_Test_idx, size=split, replace=False)
       # print(len(validation_idx))
        # Save Validation Set idx
        np.savetxt("validation_idx_%s%s" % (opt.Try,'.txt'), validation_idx.astype(int), fmt='%i', delimiter=",")
        tmp_test_idx = list(set(validation_Test_idx) - set(validation_idx))
        test_idx = np.random.choice(validation_Test_idx, size=len(tmp_test_idx), replace=False)
       # print(len(test_idx))
        #Save test idx
        np.savetxt("Test_idx_%s%s" % (opt.Try,'.txt'), test_idx, fmt='%i', delimiter=",")

    #----------
    train_sampler = SubsetRandomSampler(train_idx.astype(np.int32))
    # validation Set
    validation_sampler = SubsetRandomSampler(validation_idx.astype(np.int32))
    # Test set
    test_sampler = SubsetRandomSampler(test_idx.astype(np.int32))

    train_set = DataLoader(ds,
                           batch_size=opt.batch_size,
                           shuffle=False,
                           sampler=train_sampler,
                           num_workers=opt.num_workers)
    val_set = DataLoader(ds,
                         batch_size=opt.batch_size,
                         shuffle=False,
                         sampler=validation_sampler)
    test_set = DataLoader(ds,
                          batch_size=opt.batch_size,
                          shuffle=False,
                          sampler=test_sampler)


    num_classes = [opt.numctg,opt.numattri] #temporary lets put the number of class []
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


    print("get weight attr img...")

    Weight_attribute = get_weight_attr_img(opt)
    print(len(Weight_attribute))
    # define loss function
    criterion_softmax = nn.CrossEntropyLoss(weight=opt.loss_weight)
    criterion_binary = torch.nn.BCELoss(Weight_attribute)


    # use cuda
    if opt.cuda:
        model = model.cuda(opt.devices[0])
        criterion_softmax = criterion_softmax.cuda(opt.devices[0])
        criterion_binary = criterion_binary.cuda(opt.devices[0])
        #cudnn.benchmark = True

    # Train model
    if opt.mode == "Train":
        train(model, criterion_softmax, criterion_binary, train_set, val_set, opt)
    # Test model
    elif opt.mode == "Test":
        test(model, criterion_softmax, criterion_binary, test_set, opt)


if __name__ == "__main__":
    main()
