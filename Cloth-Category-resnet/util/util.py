import os
import copy
import numpy as np
import logging
import collections
from PIL import Image


def tensor2im(image_tensor, mean, std, imtype=np.uint8):
    image_numpy = image_tensor.cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = image_numpy.transpose(1, 2, 0)
    image_numpy *= std
    image_numpy += mean
    image_numpy *= 255.0
    return image_numpy.astype(imtype)

def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def rmdir(path):
    if os.path.exists(path):
        os.system('rm -rf ' + path)

def print_loss(loss_list, label, epoch=0, batch_iter=0):
    if label == "Test":
        f = open('Test_Loss.txt', 'a')
        f.write("%f %f" % (loss_list[0], loss_list[1]) + '\n')
        f.close()
        logging.info("[ %s Loss ] of Test Dataset:" % (label))
    else:
        logging.info("[ %s Loss ] of Test Dataset:" % (label))
        logging.info("[ %s Loss ] of Epoch %d Batch %d" % (label, epoch, batch_iter))

    if label=="Train":
         f = open('Train_Loss.txt', 'a')
         f.write("%f %f" % (loss_list[0],loss_list[1])+'\n')
         f.close()


    if label=="Validate":
        f = open('Validate_Loss.txt', 'a')
        f.write("%f %f" % (loss_list[0], loss_list[1]) + '\n')
        f.close()
    
    for index, loss in enumerate(loss_list):
        logging.info("----Categoty-Attribute %d:  %f" %(index, loss))

def print_accuracy_soft(accuracy_list, label, epoch=0, batch_iter=0):
    if label == "Test":
        f = open('Test_TopK_Acc_Ctg.txt', 'a')
        f.write("%f %f" % (accuracy_list[0], accuracy_list[1]) + '\n')
        f.close()
        logging.info("[ %s Accu ] of Test Dataset:" % (label))

    else:
        logging.info("[ %s Accu ] of Epoch %d Batch %d" %(label, epoch, batch_iter))
    
    for index, item in enumerate(accuracy_list):
        logging.info("----Category %d: %f" % (index, accuracy_list[index]))

    if label=="Train":
         f = open('Train_Topk_Acc_Ctg.txt', 'a')
         f.write("%f %f" % (accuracy_list[0],accuracy_list[1])+'\n')
         f.close()


    if label=="Validate":
        f = open('Val_Topk_Acc_Ctg.txt', 'a')
        f.write("%f %f" % (accuracy_list[0], accuracy_list[1]) + '\n')
        f.close()

def print_accuracy_bin(accuracy_list, label, epoch=0, batch_iter=0):
    if label == "Test":
        f = open('Test_TopK_Acc_Attr.txt', 'a')
        for item in accuracy_list:
            for i in range(len(item)):
                f.write("%f  " % (item[i]))
        f.write('\n')
        f.close()
        logging.info("[ %s Accu ] of Test Dataset:" % (label))
    else:
        logging.info("[ %s Accu ] of Epoch %d Batch %d" % (label, epoch, batch_iter))

    for i in range(accuracy_list.size(0)):
       for j in range(accuracy_list.size(1)):
          logging.info("----Attribute %d Top%d: %f" % (j, i+1,accuracy_list[i,j]))

    if label=="Train":
        f = open('Train_TopK_Acc_Attr.txt', 'a')
        for item in accuracy_list:
            for i in range(len(item)):
                f.write("%f  " % (item[i]))
        f.write('\n')
        f.close()


    if label=="Validate":
        f = open('Val_TopK_Acc_Attr.txt', 'a')
        for item in accuracy_list:
            for i in range(len(item)):
                f.write("%f  " % (item[i]))
        f.write('\n')
        f.close()

def opt2file(opt, dst_file):
    args = vars(opt) 
    with open(dst_file, 'wt') as opt_file:
        opt_file.write('------------ Options -------------\n')
        print '------------ Options -------------'
        for k, v in sorted(args.items()):
            opt_file.write('%s: %s\n' % (str(k), str(v)))
            print "%s: %s" %(str(k), str(v))
        opt_file.write('-------------- End ----------------\n')
        print '-------------- End ----------------'

