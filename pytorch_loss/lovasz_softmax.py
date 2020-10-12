#!/usr/bin/python
# -*- encoding: utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.amp as amp



##
# version 1: use torch.autograd
class LovaszSoftmax(nn.Module):
    '''
    This is the autograd version
    '''
    def __init__(self, reduction='mean', ignore_index=-100):
        super(LovaszSoftmax, self).__init__()
        self.reduction = reduction
        self.lb_ignore = ignore_index

    def forward(self, logits, label):
        '''
        args: logits: tensor of shape (N, C, H, W)
        args: label: tensor of shape(N, H, W)
        '''
        # overcome ignored label
        n, c, h, w = logits.size()
        logits = logits.transpose(0, 1).reshape(c, -1).float() # use fp32 to avoid nan
        label = label.view(-1)

        idx = label.ne(self.lb_ignore).nonzero(as_tuple=False).squeeze()
        probs = logits.softmax(dim=0)[:, idx]
        label = label[idx]
        lb_one_hot = torch.zeros_like(probs).scatter_(
                0, label.unsqueeze(0), 1).detach()

        errs = (lb_one_hot - probs).abs()
        errs_sort, errs_order = torch.sort(errs, dim=1, descending=True)
        n_samples = errs.size(1)

        # lovasz extension grad
        with torch.no_grad():
            lb_one_hot_sort = lb_one_hot[
                torch.arange(c).unsqueeze(1).repeat(1, n_samples), errs_order
                ].detach()
            n_pos = lb_one_hot_sort.sum(dim=1, keepdim=True)
            inter = n_pos - lb_one_hot_sort.cumsum(dim=1)
            union = n_pos + (1. - lb_one_hot_sort).cumsum(dim=1)
            jacc = 1. - inter / union
            if n_samples > 1:
                jacc[:, 1:] = jacc[:, 1:] - jacc[:, :-1]

        losses = torch.einsum('ab,ab->a', errs_sort, jacc)
        if self.reduction == 'sum':
            losses = losses.sum()
        elif self.reduction == 'mean':
            losses = losses.mean()
        return losses






if __name__ == '__main__':
    import torchvision
    import torch
    import numpy as np
    import random
    torch.manual_seed(15)
    random.seed(15)
    np.random.seed(15)
    torch.backends.cudnn.deterministic = True

    class Model(nn.Module):
        def __init__(self):
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
            self.fc = nn.Conv2d(512, 19, 3, 1, 1)
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
            out = F.interpolate(feat, x.size()[2:], mode='bilinear', align_corners=True)
            return out

    net1 = Model()
    net2 = Model()
    net2.load_state_dict(net1.state_dict())
    red = 'mean'
    criteria1 = LabelSmoothSoftmaxCEV2(lb_smooth=0.1, ignore_index=255, reduction=red)
    criteria2 = LabelSmoothSoftmaxCEV1(lb_smooth=0.1, ignore_index=255, reduction=red)
    net1.cuda()
    net2.cuda()
    net1.train()
    net2.train()
    criteria1.cuda()
    criteria2.cuda()

    optim1 = torch.optim.SGD(net1.parameters(), lr=1e-2)
    optim2 = torch.optim.SGD(net2.parameters(), lr=1e-2)

    bs = 64
    for it in range(300):
        inten = torch.randn(bs, 3, 224, 224).cuda()
        lbs = torch.randint(0, 19, (bs, 224, 224)).cuda()
        lbs[1, 1, 1] = 255
        lbs[30, 3, 2:200] = 255
        lbs[18, 4:7, 8:200] = 255
        logits = net1(inten)
        loss1 = criteria1(logits, lbs)
        optim1.zero_grad()
        loss1.backward()
        optim1.step()
        #  print(net1.fc.weight[:, :5])
        logits = net2(inten)
        loss2 = criteria2(logits, lbs)
        optim2.zero_grad()
        loss2.backward()
        optim2.step()
        #  net1.load_state_dict(net2.state_dict())
        #  print(net2.fc.weight[:, :5])
        with torch.no_grad():
            if (it+1) % 50 == 0:
                print('iter: {}, ================='.format(it+1))
                #  print(net1.fc.weight.numel())
                print('fc weight: ', torch.mean(torch.abs(net1.fc.weight - net2.fc.weight)).item())

                print('conv1 weight: ', torch.mean(torch.abs(net1.conv1.weight - net2.conv1.weight)).item())
                print('loss: ', loss1.item() - loss2.item())
