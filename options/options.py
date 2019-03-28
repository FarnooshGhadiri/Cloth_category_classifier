import os
import torch
import argparse

class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        
        self.parser.add_argument('--data_dir', default='/network/tmp1/ghadirif/DeepFashion', help='path to the data directory containing Img and annotation')
        #self.parser.add_argument('--results_dir', default='/network/tmp1/ghadirif/DeepFashion', help='path to the data directory containing Img and annotation')        
        self.parser.add_argument('--name', default='my_experiment', help='subdirectory name for training or testing, snapshot, splited dataset and test results exist here')
        self.parser.add_argument('--mode', default='Train', help='run mode of training or testing. [Train | Test | train | test]')
        self.parser.add_argument('--img_size', type=int, default=256, help='scale image to the size prepared for croping')
        self.parser.add_argument('--crop_size', type=int, default=224, help='then crop image to the size as network input')
        self.parser.add_argument('--batch_size', type=int, default=64, help='batch size of network input. Note that batch_size should only set to 1 in Test mode')
        self.parser.add_argument('--num_workers', type=int, default=4, help='how many subprocesses to use for data loading')
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--pos_weights', action='store_true', help='If true, use pos_weights with BCELossWithLogits')



        self.parser.add_argument('--checkpoint_name', type=str, default="", help='path to pretrained model or model to deploy')
        self.parser.add_argument('--pretrain', action='store_true', help='default false. If true, load pretrained model to initizaize model state_dict')
        ## for train

        self.parser.add_argument('--numctg', type=int, default=50, help='Number of cloth categories')
        self.parser.add_argument('--numattri', type=int, default=1000, help='Number of attribute')
        self.parser.add_argument('--epochs', type=int, default=100, help='number of epochs for training')
        self.parser.add_argument('--save_epoch_freq', type=int, default=1, help='save snapshot every $save_epoch_freq epoches training')
        self.parser.add_argument('--save_batch_iter_freq', type=int, default=1000, help='save snapshot every $save_batch_iter_freq training') 
        self.parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate')
        self.parser.add_argument('--gamma', type=float, default=0.1, help='multiplicative factor of learning rate decay.')
        self.parser.add_argument('--lr_mult_w', type=float, default=20, help='learning rate of W of last layer parameter will be lr*lr_mult_w')
        self.parser.add_argument('--lr_mult_b', type=float, default=20, help='learning rate of b of last layer parameter will be lr*lr_mult_b')
        self.parser.add_argument('--lr_policy', type=str, default='step', help='learning rate policy: lambda|step|plateau')
        self.parser.add_argument('--lr_decay_in_epoch', type=int, default=1, help='multiply by a gamma every lr_decay_in_epoch iterations')
        self.parser.add_argument('--momentum', type=float, default=0.9, help='momentum of SGD')
        self.parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay of SGD')
        self.parser.add_argument('--loss_weight', type=str, default='', help='list. Loss weight for cross entropy loss.For example set $loss_weight to [1, 0.8, 0.8] for a 3 labels classification')

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
        
        # mode
        if opt.mode not in ["Train", "Test", "train", "test"]:
            raise Exception("cannot recognize flag `mode`")
        opt.mode = opt.mode.capitalize()
        if opt.mode == "Test":
            opt.shuffle = False

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
        if opt.loss_weight == "":
            opt.loss_weight=None
        else:
            opt.loss_weight = torch.FloatTensor(eval(opt.loss_weight))

        return opt

if __name__ == "__main__":
    op = Options()
    op.parse()
