import os
import torch
import argparse

class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        
        self.parser.add_argument('--data_dir', default='/network/tmp1/ghadirif/DeepFashion', help='path to the data directory containing Img and annotation')
        #self.parser.add_argument('--results_dir', default='/network/tmp1/ghadirif/DeepFashion', help='path to the data directory containing Img and annotation')        
        self.parser.add_argument('--name', default='my_experiment', help='subdirectory name for training or testing, snapshot, splited dataset and test results exist here')
        self.parser.add_argument('--mode', default='train', choices=['train', 'validate', 'test'])
        self.parser.add_argument('--resnet_type', default='resnet18', choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']) 
        self.parser.add_argument('--img_size', type=int, default=256, help='scale image to the size prepared for croping')
        self.parser.add_argument('--crop_size', type=int, default=224, help='then crop image to the size as network input')
        self.parser.add_argument('--batch_size', type=int, default=64, help='batch size of network input. Note that batch_size should only set to 1 in Test mode')
        self.parser.add_argument('--num_workers', type=int, default=4, help='how many subprocesses to use for data loading')
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--loss', type=str, default='bce,', choices=['bce', 'hinge'], help='Loss function')
        self.parser.add_argument('--use_pretrain', action='store_true', help='If set, initialise resnet with pre-trained ImageNet weights')
        self.parser.add_argument('--local_features', action='store_true', help='')
        self.parser.add_argument('--beta', type=float, default=1.0)
        
        self.parser.add_argument('--reduce_sum', action='store_true', help='If true, BCE loss is sum then mean')
        self.parser.add_argument('--pos_weights', action='store_true', help='If true, use pos_weights with BCELossWithLogits')
        self.parser.add_argument('--pos_weights_scale', type=float, default=1.0, help='weights / (max(weights) / k)') # in other words, specifiying k means the max(wts) will be == k
        
        self.parser.add_argument('--save_every', type=int, default=1, help='Save a checkpoint every this many epochs')

        self.parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint. If set to `auto`, it will try find the most recent .pth in the model directory.')

        self.parser.add_argument('--num_ctg', type=int, default=50, help='Number of cloth categories')
        self.parser.add_argument('--num_attr', type=int, default=1000, help='Number of attribute')
        
        self.parser.add_argument('--epochs', type=int, default=100, help='number of epochs for training')


        self.parser.add_argument('--optimizer', type=str, default='adam', choices=['sgd', 'adam'])
        self.parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate')
        self.parser.add_argument('--gamma', type=float, default=0.1, help='multiplicative factor of learning rate decay.')

        ## for test
        self.parser.add_argument('--top_k', type=str, default='(3,5)', help='tuple. We only take top k classification results into accuracy consideration')
        self.parser.add_argument('--score_thres', type=str, default='0.1', help='float or list. We only take classification results whose score is bigger than score_thres into recall consideration')

        
        # these tow param below used only in deploy.py
        self.parser.add_argument('--label_file', type=str, default="", help='label file only for deploy a checkpoint model')
        self.parser.add_argument('--classify_dir', type=str, default="", help='directory where data.txt to be classified exists')
        
    
        self.parser.add_argument('--display_validate_freq', type=int, default=1, help='test validate dateset every $validate_freq batches iteration')
        self.parser.add_argument('--display_data_freq', type=int, default=1, help='frequency of showing training data on web browser')
        self.parser.add_argument('--display_image_ratio', type=float, default=2.0, help='ratio of images in a batch showing on web browser')

    def parse(self):
        opt = self.parser.parse_args()

        # devices id
        gpu_ids = opt.gpu_ids.split(',')
        opt.devices = []
        for id in gpu_ids:
            if eval(id) >= 0:
                opt.devices.append(eval(id))
        # cuda
        opt.cuda = False
        if len(opt.devices) > 0 and torch.cuda.is_available():
            opt.cuda = True

        opt.top_k = eval(opt.top_k)

        return opt

if __name__ == "__main__":
    op = Options()
    op.parse()
