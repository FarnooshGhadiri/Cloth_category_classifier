import os
import torch
import argparse

class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        
        self.parser.add_argument('--data_dir', default='/network/tmp1/ghadirif/DeepFashion',
                                 help="Path to the directory which contains the 'Img' and 'Anno' folders.")
        self.parser.add_argument('--name', default='my_experiment',
                                 help="Name of your experiment")
        self.parser.add_argument('--mode', default='train',
                                 choices=['train', 'validate', 'test'])
        self.parser.add_argument('--resnet_type', default='resnet18',
                                 choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'],
                                 help="What resnet is to be used as the backbone")
        self.parser.add_argument('--img_size', type=int, default=256,
                                 help="Scale image to this size before cropping")
        self.parser.add_argument('--crop_size', type=int, default=224,
                                 help="Randomly crop the scaled image to this size")
        self.parser.add_argument('--batch_size', type=int, default=64)
        self.parser.add_argument('--num_workers', type=int, default=4,
                                 help="How many subprocesses to use for data loading")
        self.parser.add_argument('--gpu_ids', type=str, default='0',
                                 help="""Comma-separated numbers denoting gpu ids,
                                 e.g. 0  0,1,2, 0,2. use -1 for CPU""")
        self.parser.add_argument('--loss', type=str, default='bce',
                                 choices=['bce', 'hinge'],
                                 help='Loss function for attribute prediction')
        
        #self.parser.add_argument('--use_pretrain', action='store_true', help='If set, initialise resnet with pre-trained ImageNet weights')
        
        self.parser.add_argument('--beta', type=float, default=1.0,
                                 help="""Coefficient to weight bounding box prediction.
                                 If set to 0, then bbox prediction will be disabled.""")
        self.parser.add_argument('--data_aug', action='store_true',
                                 help="Use data augmentation (this should always be set!!!)?")
        self.parser.add_argument('--fp16', action='store_true',
                                 help="""Do float16 computation? (Requires nvidia apex library)
                                 Apart from giving GPU memory savings, this may also speed up
                                 training.""")
        
        self.parser.add_argument('--reduce_sum', action='store_true', help='If true, BCE loss is sum then mean')

        # ** EXPERIMENTAL OPTIONS **
        self.parser.add_argument('--pos_weights', action='store_true', help='If true, use pos_weights with BCELossWithLogits')
        self.parser.add_argument('--pos_weights_scale', type=float, default=1.0, help='weights / (max(weights) / k)') # in other words, specifiying k means the max(wts) will be == k
        # **************************
        
        self.parser.add_argument('--save_every', type=int, default=1,
                                 help='Save a checkpoint every this many epochs')
        self.parser.add_argument('--resume', type=str, default=None,
                                 help="""Path to checkpoint. If set to `auto`, it will try find the 
                                 most recent .pth in the model directory.""")
        self.parser.add_argument('--epochs', type=int, default=1000,
                                 help="Number of epochs to train for")
        self.parser.add_argument('--optimizer', type=str, default='adam', choices=['sgd', 'adam'],
                                 help="What optimizer to use. (Highly recommended to use ADAM)")
        self.parser.add_argument('--lr', type=float, default=1e-4,
                                 help="Learning rate")
        self.parser.add_argument('--eps', type=float, default=1e-8,
                                 help="""Epsilon (only applies to ADAM). If you are experiencing spikes
                                 in training, it may be because of numeric instabilities in the optimization,
                                 in which case this number should be bigger, e.g. 1e-4.""")
        self.parser.add_argument('--top_k', type=str, default='3,5',
                                 help="""The k values that we should compute top-k for. For example, the default
                                 value is '3,5', which means we compute our accuracy metrics for top-3 and top-k.""")

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

        #opt.top_k = eval(opt.top_k)
        opt.top_k = [int(x) for x in opt.top_k.split(',')]

        return opt

if __name__ == "__main__":
    op = Options()
    op.parse()
