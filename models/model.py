import os
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch
from torch import nn
from torchvision.models import resnet
import numpy as np

class FashionResnet(nn.Module):

    def __init__(self,
                 num_categories,
                 num_attrs,
                 resnet_type='resnet18',
                 use_pretrain=False):
        """
        num_softmax: number of classes for discrete classification
        num_binary: number of classes for binary classification
        """
        super(FashionResnet, self).__init__()
        net_ = getattr(resnet, resnet_type)(pretrained=use_pretrain)
        self.use_pretrain = use_pretrain
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
        if self.use_pretrain:
            x = x*0.5 + 0.5 # denorm
            # subtract mean
            x[:, 0] -= 0.485
            x[:, 1] -= 0.456
            x[:, 2] -= 0.406
            x[:, 0] /= 0.229
            x[:, 1] /= 0.224
            x[:, 2] /= 0.225
        features = self.basemodel(x)
        features = features.view(-1, self.output_dim)
        out_bin = self.fc_bin(features)
        out_cls = self.fc_cls(features)
        return out_cls, out_bin


class FashionResnetLG(nn.Module):
    def __init__(self,
                 num_categories,
                 num_attrs,
                 resnet_type='resnet18',
                 use_pretrain=False):
        """
        num_softmax: number of classes for discrete classification
        num_binary: number of classes for binary classification
        """
        super(FashionResnetLG, self).__init__()
        net_ = getattr(resnet, resnet_type)(pretrained=use_pretrain)
        self.use_pretrain = use_pretrain
        #net_.avgpool = nn.AdaptiveAvgPool2d(1)
        self.num_categories = num_categories
        self.num_attrs = num_attrs
        self.basemodel = torch.nn.Sequential(*list(net_.children())[:-2])
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if resnet_type in ['resnet18', 'resnet34']:
            self.output_nf = 512
        else:
            self.output_nf = 2048
        self.spatial_dim = 7
        nf_bottleneck = self.output_nf + (7*7) # assuming 224px image
        self.fc_bin = nn.Linear(nf_bottleneck, self.num_attrs)
        self.fc_cls = nn.Linear(nf_bottleneck, self.num_categories)

    def forward(self, x):
        if self.use_pretrain:
            x = x*0.5 + 0.5 # denorm
            # subtract mean
            x[:, 0] -= 0.485
            x[:, 1] -= 0.456
            x[:, 2] -= 0.406
            x[:, 0] /= 0.229
            x[:, 1] /= 0.224
            x[:, 2] /= 0.225
        features = self.basemodel(x)
        features_global = self.avg_pool(features).view(-1, self.output_nf)
        features_local = features.mean(dim=1).reshape(-1, self.spatial_dim**2)
        new_features = torch.cat((features_global, features_local), dim=1)
        out_bin = self.fc_bin(new_features)
        out_cls = self.fc_cls(new_features)
        return out_cls, out_bin


class FashionResnet2L(nn.Module):
    def __init__(self,
                 num_categories,
                 num_attrs,
                 resnet_type='resnet18',
                 use_pretrain=False):
        super(FashionResnet2L, self).__init__()
        net_ = getattr(resnet, resnet_type)()
        self.num_categories = num_categories
        self.num_attrs = num_attrs
        self.basemodel = torch.nn.Sequential(*list(net_.children())[:-2])
        self.keys = self.basemodel._modules.keys()
        # Save the keys of the last two resblocks,
        # which correspond to spatial dims 7x7 and 14x14
        self.selected_keys = list(self.keys)[-2:]
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if resnet_type in ['resnet18', 'resnet34']:
            self.output_nf = 512
        else:
            self.output_nf = 2048
        self.spatial_dim = 7
        nf_bottleneck = self.output_nf + (7*7) + (14*14) # assuming 224px image
        self.fc_bin = nn.Linear(nf_bottleneck, self.num_attrs)
        self.fc_cls = nn.Linear(nf_bottleneck, self.num_categories)

    def forward(self, input):
        x = input
        buf = []
        for key in self.keys:
            x = self.basemodel._modules[key](x)
            if key in self.selected_keys:
                buf.append(x)
        buf_pooled = []
        for elem in buf:
            elem = elem.mean(dim=1)
            elem = elem.reshape( -1, np.prod(list(elem.size()[1:])) )
            buf_pooled.append(elem)
        local_features = torch.cat(buf_pooled, dim=1)
        global_features = self.avg_pool(x).view(-1, self.output_nf)
        all_features = torch.cat((local_features, global_features), dim=1)
        return self.fc_cls(all_features), self.fc_bin(all_features)
