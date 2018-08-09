import torch
from torch import nn
from torchvision.models import resnet
from torch.autograd import Variable

class FashionNet(nn.Module):
    """
    Take a ResNet18 model, chop off the classification layer,
      and then add two classification layers of our own (one
      a softmax and the other a binary one).
    """
    def __init__(self, num_softmax, num_binary):
        """
        num_softmax: number of classes for discrete classification
        num_binary: number of classes for binary classification
        """
        super(FashionNet, self).__init__()
        net_ = resnet.resnet18()
        self.base = torch.nn.Sequential(*list(net_.children())[:-1])
        self.fc_softmax = nn.Linear(512, num_softmax)
        self.fc_binary = nn.Linear(512, num_binary)
    def forward(self, x):
        features = self.base(x)
        out=self.fc_softmax(features)
        out2=self.fc_binary(features)
        return out, out2

net = FashionNet(10, 100)
x = torch.FloatTensor((4, 3, 224, 224))
x1 = Variable(x, requires_grad=True)
net(x1)

print(net)