import os
import sys
import time
import copy
import random
import logging
import numpy as np
import torch
from operator import add
print "Pytorch Version: ", torch.__version__
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
from data.input import get_attr_name, get_Ctg_name

#approach like this
#  https://www.ritchievink.com/blog/2018/04/12/transfer-learning-with-pytorch-assessing-road-safety-with-computer-vision/

def forward_batch(model, criterion_softmax, criterion_binary, inputs, target_softmax,target_binary, opt, phase):
    if opt.cuda:
        inputs = inputs.cuda(opt.devices[0], async=True)

    if phase in ["Train"]:
        inputs_var = Variable(inputs, requires_grad=True)
        # logging.info("Switch to Train Mode")
        model.train()
    elif phase in ["Validate", "Test"]:
         with torch.no_grad():
            inputs_var = Variable(inputs, volatile=True)
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
        #print(output_softmax,output_binary)

    # Calculate loss for sigmoid
    if opt.cuda:
        target_binary = target_binary.cuda(opt.devices[0], async=True)
    bin_loss = criterion_binary(output_binary, target_binary)

    # calculate loss for softmax

    if opt.cuda:
         target_softmax = target_softmax.cuda(opt.devices[0], async=True)

    cls_loss = criterion_softmax(output_softmax, target_softmax)

    return output_binary, output_softmax, bin_loss, cls_loss



def forward_dataset(model, criterion_softmax, criterion_binary, data_loader, opt):
    sum_batch = 0
    avg_accuracy = torch.zeros(len(opt.top_k))
    avg_accuracy_bin = torch.zeros(len(opt.top_k),6)
    avg_loss = [0,0]
    for i, data in enumerate(data_loader):
        #print(len(data_loader))
        if opt.mode == "Test":
            logging.info("test %s/%s image" % (i, len(data_loader)))

        sum_batch += 1
        inputs, target_softmax,target_binary = data
        output_binary, output_softmax, bin_loss, cls_loss = forward_batch(model, criterion_softmax, criterion_binary, inputs, target_softmax,target_binary, opt, "Validate")
        acc_softmax,acc_binary = calc_accuracy(output_softmax,output_binary, target_softmax,target_binary, opt.score_thres, opt.top_k,opt)
        # accumulate accuracy
        for k in range(len(opt.top_k)):
            avg_accuracy[k] = acc_softmax[k] + avg_accuracy[k]

        avg_accuracy_bin = avg_accuracy_bin + acc_binary
        avg_loss = list( map(add, avg_loss, [bin_loss,cls_loss]))

    # average on batches
    avg_accuracy /=float(sum_batch)
    avg_accuracy_bin /= float(sum_batch)
    avg_loss = map(lambda x: x/(sum_batch), avg_loss)
    return avg_accuracy, avg_accuracy_bin, avg_loss


def calc_accuracy(output_binary, output_softmax, target_softmax,target_binary, score_thres, top_k,opt):
    #Softmax accuracy
    max_k = max(top_k)
    batch_size=target_softmax.size(0)
    _, pred = output_softmax.topk(max_k, 1, True, True)
    pred=pred.t()
    correct = pred.eq(target_softmax.view(1, -1).expand_as(pred))
    acc_soft=[]
    for k in top_k:
        correct_k = correct[:k].view(-1).float().sum(0)
        acc_soft.append(correct_k.mul_(100.0/batch_size))

    #Binary accuracy
    _,attr_type = get_attr_name(opt)
    # print(attr_type)

    acc_bin=torch.zeros(len(top_k),6)
    k_num=0
    for k in top_k:
      correct = 0
      correct_attr = []
      correct_attr_type = torch.zeros(5)
      tmp_correct_attr = torch.zeros(5)
      _, pred = output_binary.topk(k, 1, True, True)
      pred = pred.t()
      for b in range(batch_size):
          correct_attr=[]
          tmp_target = (target_binary[b,:] == 1).nonzero()
          tmp_correct = np.intersect1d(pred[:,b], tmp_target)
          if len(tmp_correct)<>0:
              correct_attr = attr_type[tmp_correct]
              tmp_correct_attr[0] = (correct_attr == 1).sum()
              tmp_correct_attr[1] = (correct_attr == 2).sum()
              tmp_correct_attr[2] = (correct_attr == 3).sum()
              tmp_correct_attr[3] = (correct_attr == 4).sum()
              tmp_correct_attr[4] = (correct_attr == 5).sum()



          correct = correct + len(tmp_correct)
          correct_attr_type[0] = correct_attr_type[0] + tmp_correct_attr[0]
          correct_attr_type[1] = correct_attr_type[1] + tmp_correct_attr[1]
          correct_attr_type[2] = correct_attr_type[2] + tmp_correct_attr[2]
          correct_attr_type[3] = correct_attr_type[3] + tmp_correct_attr[3]
          correct_attr_type[4] = correct_attr_type[4] + tmp_correct_attr[4]

      acc_bin[k_num,0] = (correct*(100.0/batch_size))
      acc_bin[k_num,1] = (correct_attr_type[0]*(100.0 / batch_size))
      acc_bin[k_num,2] = (correct_attr_type[1]*(100.0 / batch_size))
      acc_bin[k_num,3] = (correct_attr_type[2]*(100.0 / batch_size))
      acc_bin[k_num,4] = (correct_attr_type[3]*(100.0 / batch_size))
      acc_bin[k_num,5] = (correct_attr_type[4]*(100.0 / batch_size))
      k_num = k_num + 1

    return acc_soft, acc_bin


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
        for i, data in enumerate(train_set):
            inputs, target_softmax,target_binary = data
            output_binary, output_softmax, bin_loss, cls_loss = forward_batch(model, criterion_softmax, criterion_binary, inputs, target_softmax,target_binary, opt, "Train")
            optimizer.zero_grad()
            loss = bin_loss + cls_loss
            loss_list = [bin_loss[0],cls_loss[0]]
            loss.backward()
            optimizer.step()

           # webvis.reset()
            epoch_batch_iter += 1
            total_batch_iter += 1

            # display train loss and accuracy
            if total_batch_iter % opt.display_train_freq == 0:
                # accuracy
                batch_accuracy_soft,batch_accuracy_bin = calc_accuracy(output_binary, output_softmax, target_softmax,target_binary, opt.score_thres, opt.top_k,opt)
                util.print_loss(loss_list, "Train", epoch, total_batch_iter)
                util.print_accuracy_soft(batch_accuracy_soft, "Train", epoch, total_batch_iter)
                util.print_accuracy_bin(batch_accuracy_bin, "Train", epoch, total_batch_iter)

                # if opt.display_id > 0:
                #     x_axis = epoch + float(epoch_batch_iter) / train_batch_num
                #     # TODO support accuracy visualization of multiple top_k
                #     plot_accuracy = [batch_accuracy_soft[i][opt.top_k[0]] for i in range(len(batch_accuracy_soft))]
                #     accuracy_list = [item["ratio"] for item in plot_accuracy]
                  #  webvis.plot_points(x_axis, loss_list, "Loss", "Train")
                   # webvis.plot_points(x_axis, accuracy_list, "Accuracy", "Train")

            # display train data
            # if total_batch_iter % opt.display_data_freq == 0:
            #     image_list = list()
            #     show_image_num = int(np.ceil(opt.display_image_ratio * inputs.size()[0]))
            #     for index in range(show_image_num):
            #         input_im = util.tensor2im(inputs[index], opt.mean, opt.std)
            #         class_label = "Image_" + str(index)
            #
            #         image_list.append((class_label, input_im))
            #     image_dict = OrderedDict(image_list)
            #     save_result = total_batch_iter % opt.update_html_freq
            #    # webvis.plot_images(image_dict, opt.display_id + 2 * opt.class_num, epoch, save_result)

            # validate and display validate loss and accuracy
            if len(val_set) > 0:
                avg_val_accuracy, avg_val_accuracy_bin, avg_val_loss = validate(model, criterion_softmax, criterion_binary, val_set, opt)
                util.print_loss(avg_val_loss, "Validate", epoch, total_batch_iter)
                util.print_accuracy_soft(avg_val_accuracy, "Validate", epoch, total_batch_iter)
                util.print_accuracy_bin(avg_val_accuracy_bin, "Validate", epoch, total_batch_iter)
               # if opt.display_id > 0:
                  #  webvis.plot_points(x_axis, val_loss, "Loss", "Validate")
                  # webvis.plot_points(x_axis, accuracy_list, "Accuracy", "Validate")

            # save snapshot
            if total_batch_iter % opt.save_batch_iter_freq == 0:
                logging.info("saving the latest model (epoch %d, total_batch_iter %d)" % (epoch, total_batch_iter))
                save_model(model, opt, epoch)
                # TODO snapshot loss and accuracy

      #  logging.info('End of epoch %d / %d \t Time Taken: %d sec' %
                    # (epoch, opt.sum_epoch, time.time() - epoch_start_t))

            if epoch % opt.save_epoch_freq == 0:
               logging.info('saving the model at the end of epoch %d, iters %d' % (epoch + 1, total_batch_iter))
               save_model(model, opt, epoch + 1)

            # adjust learning rate
        logging.info('learning rate = %.7f epoch = %d' % (lr, epoch))
    logging.info("--------Optimization Done--------")


def validate(model, criterion_softmax, criterion_binary, val_set, opt):
    return forward_dataset(model, criterion_softmax, criterion_binary, val_set, opt)


def test(model, criterion_softmax, criterion_binary, test_set, opt):
    logging.info("####################Test Model###################")
    test_accuracy, test_accuracy_bin, test_loss = forward_dataset(model, criterion_softmax, criterion_binary, test_set, opt)
    logging.info("data_dir:   " + opt.data_dir + "/TestSet/")
    logging.info("score_thres:" + str(opt.score_thres))
    for index, item in enumerate(test_accuracy):
            logging.info("----Accuracy of Top%d: %f" % (opt.top_k[index], item))

    for i in range(test_accuracy_bin.size(0)):
       for j in range(test_accuracy_bin.size(1)):
          logging.info("----Attribute %d Top%d: %f" % (j, opt.top_k[i],test_accuracy_bin[i,j]))

    for index, item in enumerate(test_loss):
          logging.info("----Loss of ctg and attrs%d: %f" % (index, item))
    logging.info("#################Finished Testing################")


def main():
    # parse options
    op = Options()
    opt = op.parse()

    # initialize train or test working dir
    trainer_dir = "trainer_" + opt.name
    opt.model_dir = os.path.join(opt.dir, trainer_dir, "Train")
    opt.data_dir = os.path.join(opt.dir, trainer_dir, "Data")
    opt.test_dir = os.path.join(opt.dir, trainer_dir, "Test")

    if not os.path.exists(opt.data_dir):
        os.makedirs(opt.data_dir)
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

    # load train or test data
    ds = DeepFashionDataset(opt)
    num_data = len(ds)
    indices = list(range(num_data))
    split = int((opt.ratio[1] + opt.ratio[2]) * num_data)
    validation_Test_idx = np.random.choice(indices, size=split, replace=False)
    train_idx = list(set(indices) - set(validation_Test_idx))
    train_sampler = SubsetRandomSampler(train_idx)
    # validation Set
    split = int(round(0.5 * len(validation_Test_idx)))
    validation_idx = np.random.choice(validation_Test_idx, size=split, replace=False)
    validation_sampler = SubsetRandomSampler(validation_idx)
    # Test set
    test_idx = list(set(validation_Test_idx) - set(validation_idx))
    test_sampler = SubsetRandomSampler(test_idx)

    train_set = DataLoader(ds, batch_size=opt.batch_size, shuffle=False, sampler=train_sampler)
    val_set = DataLoader(ds, batch_size=opt.batch_size, shuffle=False, sampler=validation_sampler)
    test_set = DataLoader(ds, batch_size=opt.batch_size, shuffle=False, sampler=test_sampler)



    num_classes = [opt.numctg,opt.numattri] #temporary lets put the number of class []
    opt.class_num = len(num_classes)

    # load model
    model = Fashion_model(opt, num_classes)
    logging.info(model)

    # imagenet pretrain model
    if opt.pretrain:
        logging.info("use imagenet pretrained model")

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



    # define loss function
    criterion_softmax = nn.CrossEntropyLoss(weight=opt.loss_weight)
    criterion_binary = torch.nn.BCELoss()


    # use cuda
    if opt.cuda:
        model = model.cuda(opt.devices[0])
        criterion_softmax = criterion_softmax.cuda(opt.devices[0])
        criterion_binary = criterion_binary.cuda(opt.devices[0])
        cudnn.benchmark = True

    # Train model
    if opt.mode == "Train":
        train(model, criterion_softmax,criterion_binary, train_set, val_set, opt)
    # Test model
    elif opt.mode == "Test":
        test(model, criterion_softmax,criterion_binary, test_set, opt)


if __name__ == "__main__":
    main()
