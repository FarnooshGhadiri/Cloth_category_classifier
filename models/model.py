import os
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch
from torch import nn
from torchvision.models import resnet

class Fashion_model(nn.Module):

    def __init__(self, num_categories, num_attrs, resnet_type='resnet18'):
        """
        num_softmax: number of classes for discrete classification
        num_binary: number of classes for binary classification
        """
        super(Fashion_model, self).__init__()
        net_ = getattr(resnet, resnet_type)()
        #net_.avgpool = nn.AdaptiveAvgPool2d(1)
        self.num_categories = num_categories
        self.num_attrs = num_attrs
        self.basemodel = torch.nn.Sequential(*list(net_.children())[:-1])
        if resnet_type in ['resnet18', 'resnet34']:
            self.output_dim = 512
        else:
            self.output_dim = 2048
        self.fc_bin = nn.Linear(self.output_dim, self.num_attrs)
        self.fc_cls = nn.Linear(self.output_dim, self.num_categories)

    def forward(self, x):
        features = self.basemodel(x)
        features = features.view(-1, self.output_dim)
        out_bin = self.fc_bin(features)
        out_cls = self.fc_cls(features)
        return out_cls, out_bin

def save_model(model, optimizer, model_dir, epoch):
    """Save model

    :param model: model state to save 
    :param optimizer: optim state to save
    :param model_dir: 
    :param epoch: epoch # to save
    :returns: 
    :rtype: 

    """
    checkpoint_name = "%s/epoch_%s.pth" % (model_dir, epoch)
    dd = {'model': model.state_dict(),
          'optim': optimizer.state_dict(),
          'epoch': epoch}
    torch.save(dd, checkpoint_name)
    #torch.save(model.cpu().state_dict(), checkpoint_name)
    #if opt.cuda and torch.cuda.is_available():
    #    model.cuda(opt.devices[0])

def load_model(model, optimizer, checkpoint):
    dd = torch.load(checkpoint)
    model.load_state_dict(dd['model'])
    if 'optim' in dd:
        optimizer.load_state_dict(dd['optim'])
    return dd['epoch']
