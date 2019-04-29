# Original source: https://github.com/i008/pytorch-deepfashion

"""
MIT License

Copyright (c) 2018 Jakub Cieslik

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import torch
from torch.nn import AdaptiveMaxPool2d

ROI_POOL_SIZE = (3,3)
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
