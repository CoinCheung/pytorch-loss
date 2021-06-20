#!/usr/bin/python
# -*- encoding: utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.amp as amp
import soft_dice_cpp # should import torch before import this


## Soft Dice Loss for binary segmentation
##
# v1: pytorch autograd
class SoftDiceLossV1(nn.Module):
    '''
    soft-dice loss, useful in binary segmentation
    '''
    def __init__(self,
                 p=1,
                 smooth=1):
        super(SoftDiceLossV1, self).__init__()
        self.p = p
        self.smooth = smooth

    def forward(self, logits, labels):
        '''
        inputs:
            logits: tensor of shape (N, H, W, ...)
            label: tensor of shape(N, H, W, ...)
        output:
            loss: tensor of shape(1, )
        '''
        probs = torch.sigmoid(logits)
        numer = (probs * labels).sum()
        denor = (probs.pow(self.p) + labels.pow(self.p)).sum()
        loss = 1. - (2 * numer + self.smooth) / (denor + self.smooth)
        return loss


##
# v2: self-derived grad formula
class SoftDiceLossV2(nn.Module):
    '''
    soft-dice loss, useful in binary segmentation
    '''
    def __init__(self,
                 p=1,
                 smooth=1):
        super(SoftDiceLossV2, self).__init__()
        self.p = p
        self.smooth = smooth

    def forward(self, logits, labels):
        '''
        inputs:
            logits: tensor of shape (N, H, W, ...)
            label: tensor of shape(N, H, W, ...)
        output:
            loss: tensor of shape(1, )
        '''
        logits = logits.view(1, -1)
        labels = labels.view(1, -1)
        loss = SoftDiceLossV2Func.apply(logits, labels, self.p, self.smooth)
        return loss


class SoftDiceLossV2Func(torch.autograd.Function):
    '''
    compute backward directly for better numeric stability
    '''
    @staticmethod
    @amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, logits, labels, p, smooth):
        '''
        inputs:
            logits: (N, L)
            labels: (N, L)
        outpus:
            loss: (N,)
        '''
        #  logits = logits.float()

        probs = torch.sigmoid(logits)
        numer = 2 * (probs * labels).sum(dim=1) + smooth
        denor = (probs.pow(p) + labels.pow(p)).sum(dim=1) + smooth
        loss = 1. - numer / denor

        ctx.vars = probs, labels, numer, denor, p, smooth
        return loss

    @staticmethod
    @amp.custom_bwd
    def backward(ctx, grad_output):
        '''
        compute gradient of soft-dice loss
        '''
        probs, labels, numer, denor, p, smooth = ctx.vars

        numer, denor = numer.view(-1, 1), denor.view(-1, 1)

        term1 = (1. - probs).mul_(2).mul_(labels).mul_(probs).div_(denor)

        term2 = probs.pow(p).mul_(1. - probs).mul_(numer).mul_(p).div_(denor.pow_(2))

        grads = term2.sub_(term1).mul_(grad_output)

        return grads, None, None, None


##
# v3: implement with cuda to save memory
class SoftDiceLossV3(nn.Module):
    '''
    soft-dice loss, useful in binary segmentation
    '''
    def __init__(self,
                 p=1,
                 smooth=1.):
        super(SoftDiceLossV3, self).__init__()
        self.p = p
        self.smooth = smooth

    def forward(self, logits, labels):
        '''
        inputs:
            logits: tensor of shape (N, H, W, ...)
            label: tensor of shape(N, H, W, ...)
        output:
            loss: tensor of shape(1, )
        '''
        logits = logits.view(1, -1)
        labels = labels.view(1, -1)
        loss = SoftDiceLossV3Func.apply(logits, labels, self.p, self.smooth)
        return loss


class SoftDiceLossV3Func(torch.autograd.Function):
    '''
    compute backward directly for better numeric stability
    '''
    @staticmethod
    @amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, logits, labels, p, smooth):
        '''
        inputs:
            logits: (N, L)
            labels: (N, L)
        outpus:
            loss: (N,)
        '''
        assert logits.size() == labels.size() and logits.dim() == 2
        loss = soft_dice_cpp.soft_dice_forward(logits, labels, p, smooth)
        ctx.vars = logits, labels, p, smooth
        return loss

    @staticmethod
    @amp.custom_bwd
    def backward(ctx, grad_output):
        '''
        compute gradient of soft-dice loss
        '''
        logits, labels, p, smooth = ctx.vars
        grads = soft_dice_cpp.soft_dice_backward(grad_output, logits, labels, p, smooth)
        return grads, None, None, None



if __name__ == '__main__':
    import torchvision
    import torch
    import numpy as np
    import random
    #  torch.manual_seed(15)
    #  random.seed(15)
    #  np.random.seed(15)
    #  torch.backends.cudnn.deterministic = True
    #  torch.cuda.set_device('cuda:1')

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
            self.out = nn.Conv2d(512, 1, 3, 1, 1)
        def forward(self, x):
            feat = self.conv1(x)
            feat = self.bn1(feat)
            feat = self.relu(feat)
            feat = self.maxpool(feat)
            feat = self.layer1(feat)
            feat = self.layer2(feat)
            feat = self.layer3(feat)
            feat = self.layer4(feat)
            feat = self.out(feat)
            out = F.interpolate(feat, x.size()[2:], mode='bilinear', align_corners=True)
            return out
    net1 = Model()
    net2 = Model()
    net2.load_state_dict(net1.state_dict())

    criteria1 = SoftDiceLossV3()
    criteria2 = SoftDiceLossV1()
    net1.cuda()
    net2.cuda()
    net1.train()
    net2.train()
    net1.double()
    net2.double()
    criteria1.cuda()
    criteria2.cuda()

    optim1 = torch.optim.SGD(net1.parameters(), lr=1e-2)
    optim2 = torch.optim.SGD(net2.parameters(), lr=1e-2)

    bs = 12
    size = 320, 320
    #  size = 229, 229
    for it in range(300000):
    #  for it in range(500):
        inten = torch.randn(bs, 3, *size).cuda()
        lbs = torch.randint(0, 2, (bs, *size)).cuda().float()
        inten = inten.double()
        lbs = lbs.double()

        logits = net1(inten).squeeze(1)
        loss1 = criteria1(logits, lbs)
        optim1.zero_grad()
        loss1.backward()
        optim1.step()
        logits = net2(inten).squeeze(1)
        loss2 = criteria2(logits, lbs)
        optim2.zero_grad()
        loss2.backward()
        optim2.step()

        with torch.no_grad():
            if (it+1) % 50 == 0:
                print('iter: {}, ================='.format(it+1))
                print('out.weight: ', torch.mean(torch.abs(net1.out.weight - net2.out.weight)).item())
                print('conv1.weight: ', torch.mean(torch.abs(net1.conv1.weight - net2.conv1.weight)).item())
                print('loss: ', loss1.item() - loss2.item())
