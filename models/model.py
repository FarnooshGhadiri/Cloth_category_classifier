import os
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch
from torch import nn
from torchvision.models import resnet

class Fashion_model(nn.Module):

    def __init__(self, opt, num_classes):
        """
        num_softmax: number of classes for discrete classification
        num_binary: number of classes for binary classification
        """
        super(Fashion_model, self).__init__()
        net_ = resnet.resnet18()
        net_.avgpool = nn.AdaptiveAvgPool2d(1)
        self.num_classes = num_classes
        self.basemodel = torch.nn.Sequential(*list(net_.children())[:-1])
        self.tmp_input = (torch.randn(1, opt.input_channel, opt.input_size, opt.input_size))
        self.tmp_output = self.basemodel(self.tmp_input)
        self.output_dim = int(self.tmp_output.size()[1])
        print(self.output_dim)
        #self.output_dim = 512
        for index, num_class in enumerate(num_classes):
            setattr(self, "FullyConnectedLayer_" + str(index), nn.Linear(self.output_dim, num_class))

    def forward(self, x):
        features = self.basemodel(x)
        nf = features.size()[1]
        features = features.view(-1, nf)
        for index, num_class in enumerate(self.num_classes):
            fun = eval("self.FullyConnectedLayer_" + str(index))
            out = fun(features)
            if index == 0:
                output_Softmax = out
            else:
                output_Sigmoid = F.sigmoid(out)


        return output_Softmax, output_Sigmoid


def save_model(model, opt, epoch):
    checkpoint_name = opt.model_dir + "/epoch_%s.pth" %(epoch)
    torch.save(model.cpu().state_dict(), checkpoint_name)
    if opt.cuda and torch.cuda.is_available():
        model.cuda(opt.devices[0])

