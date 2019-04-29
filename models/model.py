import os
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch
from torch import nn
from torchvision.models import resnet
import numpy as np
from .roi_pooling import (gated_roi_pooling,
                          adaptive_max_pool)

class FashionResnet(nn.Module):

    def __init__(self,
                 num_categories,
                 num_attrs,
                 resnet_type='resnet18'):
        super(FashionResnet, self).__init__()
        net_ = getattr(resnet, resnet_type)()
        #net_.avgpool = nn.AdaptiveAvgPool2d(1)
        self.num_categories = num_categories
        self.num_attrs = num_attrs
        self.basemodel = torch.nn.Sequential(*list(net_.children())[:-2])
        if resnet_type in ['resnet18', 'resnet34']:
            self.output_dim = 512
        else:
            self.output_dim = 2048
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.reduce_1x1 = nn.Conv2d(self.output_dim, 32, kernel_size=1, stride=1)
        
        # [x1, y1, x_offset, y_offset]
        self.fc_bbox_xy = nn.Linear(32*7*7, 2)
        self.fc_bbox_offset = nn.Linear(32*7*7, 2)
        
        self.tot_dim = self.output_dim + (3*3*self.output_dim)
    
        self.fc_bin = nn.Linear(self.tot_dim, self.num_attrs)
        self.fc_cls = nn.Linear(self.tot_dim, self.num_categories)
        
        self.SPATIAL_DIM = 7

    def forward(self, x):
        bs = x.size(0)
        base_features = self.basemodel(x)
        # Ok, compute local features using ROIPooling
        # on the predicted bounding boxes.
        pre_bbox = self.reduce_1x1(base_features)
        pre_bbox = pre_bbox.reshape(-1, 32*self.SPATIAL_DIM*self.SPATIAL_DIM)
        bbox_xy = F.sigmoid(self.fc_bbox_xy(pre_bbox))
        bbox_offset = F.sigmoid(self.fc_bbox_offset(pre_bbox))
        bbox = torch.cat( (bbox_xy, bbox_xy+bbox_offset), dim=1 )
        # The last two values of bbox are actually offsets,
        # so we need to make these absolute coordinates.
        #bbox[:, 2] = bbox[:, 2] + bbox[:, 0]
        #bbox[:, 3] = bbox[:, 3] + bbox[:, 1]
    
        batch_ids = torch.arange(0, bs).view(-1, bs).t().float()
        if x.is_cuda:
            batch_ids = batch_ids.cuda()
        bbox_new = torch.cat((batch_ids, bbox), dim=1)
        bbox_new[:, 1:] = bbox_new[:, 1:] * (self.SPATIAL_DIM-1)

        try:
            local_features = gated_roi_pooling(base_features, rois=bbox_new)
        except:
            print("DEBUG:")
            print("bbox_new:")
            print(bbox_new)
            print("bbox_xy:")
            print(bbox_xy)
            print("bbox_offset:")
            print(bbox_offset)
            raise Exception()
            
        local_features = local_features.reshape(-1, self.output_dim*3*3)
        
        # Ok, compute global features 
        global_features = self.avg_pool(base_features)
        global_features = global_features.reshape(-1, self.output_dim)
        
        all_features = torch.cat((global_features, local_features), dim=1)
        
        out_bin = self.fc_bin(all_features)
        out_cls = self.fc_cls(all_features)
        
        return out_cls, out_bin, bbox
