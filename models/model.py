import os
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch
from torch import nn
from torchvision.models import resnet
import numpy as np

ROI_POOL_SIZE = (3,3)
from torch.nn import AdaptiveMaxPool2d
def gated_roi_pooling(input, rois, size=ROI_POOL_SIZE):
    """
    Standard roi-pooling extended to accept a mask vector (gates) wich will set all activations to zero for
    correspnding features
    :param input: features (for instance  feature maps from vgg/resnet
    :param rois: rois [batch_id, x1, y1, x2, y2]
    :param gates: mask vector with shape [len(gates), 1]
    :param size: size of the pooled regions (for instance (3,3)
    :param spatial_scale:
    :return:
    """
    assert (rois.dim() == 2)
    assert (rois.size(1) == 5)
    output = []
    #rois = rois.data.float()
    num_rois = rois.size(0)

    #rois[:, 1:].mul_(spatial_scale)
    rois = rois.long()
    for i in range(num_rois):
        roi = rois[i]
        im_idx = roi[0]
        #im = input.narrow(0, im_idx, 1)[..., roi[2]:(roi[4] + 1), roi[1]:(roi[3] + 1)]
        im = input[im_idx][..., roi[2]:(roi[4] + 1), roi[1]:(roi[3] + 1)]
        mp = adaptive_max_pool(im, size)[0]
        output.append(mp)

    pooled_features = torch.cat(output, 0)

    return pooled_features.view(input.shape[0], -1, ROI_POOL_SIZE[0], ROI_POOL_SIZE[1])

def adaptive_max_pool(input, size):
    return AdaptiveMaxPool2d(size[0], size[1])(input)

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
