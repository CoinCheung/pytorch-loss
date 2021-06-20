
## TODO: test case should cover, n_class from 3 to 256, test ignore index, test speed and memory usage

from lovasz_softmax import LovaszSoftmaxV1, LovaszSoftmaxV3
from label_smooth import LabelSmoothSoftmaxCEV3
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
torch.manual_seed(15)
random.seed(15)
np.random.seed(15)
torch.backends.cudnn.deterministic = True


class Model(nn.Module):
    def __init__(self, n_classes):
        super(Model, self).__init__()
        net = torchvision.models.resnet18(pretrained=False)
        self.conv1 = net.conv1
        self.bn1 = net.bn1
        self.maxpool = net.maxpool
        self.relu = net.relu
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4
        self.fc = nn.Conv2d(512, n_classes, 3, 1, 1)
    def forward(self, x):
        feat = self.conv1(x)
        feat = self.bn1(feat)
        feat = self.relu(feat)
        feat = self.maxpool(feat)
        feat = self.layer1(feat)
        feat = self.layer2(feat)
        feat = self.layer3(feat)
        feat = self.layer4(feat)
        feat = self.fc(feat)
        #  out = F.interpolate(feat, x.size()[2:], mode='bilinear', align_corners=True)
        out = torch.mean(feat, dim=(2, 3))
        return out

c = 2
net1 = Model(c)
#  net2 = Model()
#  net2.load_state_dict(net1.state_dict())
red = 'mean'
#  criteria1 = LovaszSoftmaxV1(reduction='sum', ignore_index=255)
#  criteria1 = LovaszSoftmaxV3(reduction='sum', ignore_index=255)
criteria1 = LabelSmoothSoftmaxCEV3(reduction='sum', ignore_index=255)
print(criteria1)

net1.cuda()
#  net2.cuda()
net1.train()
#  net2.train()
criteria1.cuda()
#  criteria2.cuda()
#  net1 = net1.half()

optim1 = torch.optim.SGD(net1.parameters(), lr=1e-2)
#  optim2 = torch.optim.SGD(net2.parameters(), lr=1e-2)

bs, h, w = 2, 1000, 1000
for it in range(1000):
    inten = torch.randn(bs, 3, h, w).cuda()#.half()
    #  lbs = torch.randint(0, c, (bs, h, w)).cuda()
    lbs = torch.randint(0, c, (bs, )).cuda()
    #  lbs[1, 1, 1] = 255
    #  lbs[0, 3:100, 2:100] = 255
    #  lbs[1, 4:70, 28:200] = 255
    logits1 = net1(inten)
    logits1.retain_grad()
    loss1 = criteria1(logits1, lbs)
    optim1.zero_grad()
    loss1.backward()
    optim1.step()
    with torch.no_grad():
        if (it+1) % 50 == 0:
            print('iter: {}, ================='.format(it+1))
