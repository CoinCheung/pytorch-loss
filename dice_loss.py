#!/usr/bin/python
# -*- encoding: utf-8 -*-


import torch
import torch.nn as nn


class GeneralizedSoftDiceLoss(nn.Module):
    def __init__(self,
                 p=1,
                 smooth=1,
                 reduction='mean',
                 weight=None,
                 ignore_lb=255):
        super(GeneralizedSoftDiceLoss, self).__init__()
        self.p = p
        self.smooth = smooth
        self.reduction = reduction
        self.weight = None if weight is None else torch.tensor(weight)
        self.ignore_lb = ignore_lb

    def forward(self, logits, label):
        '''
        args: logits: tensor of shape (N, C, H, W)
        args: label: tensor of shape(N, H, W)
        '''
        # overcome ignored label
        ignore = label.data.cpu() == self.ignore_lb
        label = label.clone()
        label[ignore] = 0
        lb_one_hot = torch.zeros_like(logits).scatter_(1, label.unsqueeze(1), 1)
        ignore = ignore.nonzero()
        _, M = ignore.size()
        a, *b = ignore.chunk(M, dim=1)
        lb_one_hot[[a, torch.arange(lb_one_hot.size(1)).long(), *b]] = 0
        lb_one_hot = lb_one_hot.detach()

        # compute loss
        probs = torch.sigmoid(logits)
        numer = torch.sum((probs*lb_one_hot), dim=(2, 3))
        denom = torch.sum(probs.pow(self.p)+lb_one_hot.pow(self.p), dim=(2, 3))
        if not self.weight is None:
            numer = numer * self.weight.view(1, -1)
            denom = denom * self.weight.view(1, -1)
        numer = torch.sum(numer, dim=1)
        denom = torch.sum(denom, dim=1)
        loss = 1 - (2*numer+self.smooth)/(denom+self.smooth)

        if self.reduction == 'mean':
            loss = loss.mean()
        return loss


class BatchSoftDiceLoss(nn.Module):
    def __init__(self,
                 p=1,
                 smooth=1,
                 weight=None,
                 ignore_lb=255):
        super(BatchSoftDiceLoss, self).__init__()
        self.p = p
        self.smooth = smooth
        self.weight = None if weight is None else torch.tensor(weight)
        self.ignore_lb = ignore_lb

    def forward(self, logits, label):
        '''
        args: logits: tensor of shape (N, C, H, W)
        args: label: tensor of shape(N, H, W)
        '''
        # overcome ignored label
        ignore = label.data.cpu() == self.ignore_lb
        label = label.clone()
        label[ignore] = 0
        lb_one_hot = torch.zeros_like(logits).scatter_(1, label.unsqueeze(1), 1)
        ignore = ignore.nonzero()
        _, M = ignore.size()
        a, *b = ignore.chunk(M, dim=1)
        lb_one_hot[[a, torch.arange(lb_one_hot.size(1)).long(), *b]] = 0
        lb_one_hot = lb_one_hot.detach()

        # compute loss
        probs = torch.sigmoid(logits)
        numer = torch.sum((probs*lb_one_hot), dim=(2, 3))
        denom = torch.sum(probs.pow(self.p)+lb_one_hot.pow(self.p), dim=(2, 3))
        if not self.weight is None:
            numer = numer * self.weight.view(1, -1)
            denom = denom * self.weight.view(1, -1)
        numer = torch.sum(numer)
        denom = torch.sum(denom)
        loss = 1 - (2*numer+self.smooth)/(denom+self.smooth)
        return loss


if __name__ == '__main__':
    #  criteria = GeneralizedSoftDiceLoss()
    criteria = BatchSoftDiceLoss()
    #  logits = torch.randn(16, 19, 14, 14)
    im = torch.randn(16, 3, 14, 14)
    label = torch.randint(0, 19, (16, 14, 14)).long()
    net = torch.nn.Conv2d(3, 19, 3, 1, 1)
    print(label.dtype)
    label[2, 3, 3] = 255
    print(label.dtype)

    logits = net(im)
    loss = criteria(logits, label)
    loss.backward()
    print(loss)
