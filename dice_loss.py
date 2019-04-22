#!/usr/bin/python
# -*- encoding: utf-8 -*-


import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self,
                 p=2,
                 smooth=1,
                 reduction='mean',
                 weight=None,
                 ignore_lb=255):
        super(DiceLoss, self).__init__()
        self.p = p
        self.smooth = smooth
        self.reduction = reduction
        self.weight = torch.tensor(weight)
        self.ignore_lb = ignore_lb

    def forward(self, logits, label):
        # overcome ignored label
        ignore = label.data.cpu() == self.ignore_lb
        label[ignore] = 0
        lb_one_hot = logits.data.clone().zero_().scatter_(1, label.unsqueeze(1), 1)
        ignore = ignore.nonzero()
        _, M = ignore.size()
        a, *b = ignore.chunk(M, dim=1)
        lb_one_hot[[a, torch.zeros(lb_one_hot.size(1)), *b]] = 0

        # compute loss
        probs = torch.sigmoid(logits)
        numer = torch.sum((probs*lb_one_hot), dim=(2, 3))
        denom = torch.sum(probs.pow(self.p)+lb_one_hot.pow(self.p)).sum(dim=(2, 3))
        loss = 1 - 2*(numer+self.smooth)/(denom+self.smooth)
        if not self.weight is None:
            loss *= self.weight.view(1, -1)

        if self.reduction == 'none':
            loss = loss.sum(dim=1)
        elif self.reduction == 'mean':
            loss = loss.sum(dim=1).mean()
        return loss

