#!/usr/bin/python
# -*- encoding: utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F

import ohem_cpp
from .large_margin_softmax import LargeMarginSoftmaxV3


class OhemCELoss(nn.Module):

    def __init__(self, score_thresh, n_min=None, ignore_index=255):
        super(OhemCELoss, self).__init__()
        self.score_thresh = score_thresh
        self.ignore_lb = ignore_index
        self.n_min = n_min
        self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='mean')

    def forward(self, logits, labels):
        n_min = labels.numel() // 16 if self.n_min is None else self.n_min
        labels = ohem_cpp.score_ohem_label(logits.float(), labels,
                self.ignore_lb, self.score_thresh, n_min).detach()
        loss = self.criteria(logits, labels)
        return loss


class OhemLargeMarginLoss(nn.Module):

    def __init__(self, score_thresh, n_min=None, ignore_index=255):
        super(OhemLargeMarginLoss, self).__init__()
        self.score_thresh = score_thresh
        self.ignore_lb = ignore_index
        self.n_min = n_min
        self.criteria = LargeMarginSoftmaxV3(
                ignore_index=ignore_index, reduction='mean')

    def forward(self, logits, labels):
        n_min = labels.numel() // 16 if self.n_min is None else self.n_min
        labels = ohem_cpp.score_ohem_label(logits.float(), labels,
                self.ignore_lb, self.score_thresh, n_min).detach()
        loss = self.criteria(logits, labels)
        return loss


if __name__ == '__main__':
    criteria1 = OhemLargeMarginLoss(score_thresh=0.7, n_min=16*20*20//16).cuda()
    criteria2 = OhemCELoss(score_thresh=0.7, n_min=16*20*20//16).cuda()
    net1 = nn.Sequential(
        nn.Conv2d(3, 19, kernel_size=3, stride=2, padding=1),
    )
    net1.cuda()
    net1.train()
    net2 = nn.Sequential(
        nn.Conv2d(3, 19, kernel_size=3, stride=2, padding=1),
    )
    net2.cuda()
    net2.train()

    with torch.no_grad():
        inten = torch.randn(16, 3, 20, 20).cuda()
        lbs = torch.randint(0, 19, [16, 20, 20]).cuda()
        lbs[1, 10, 10] = 255

    torch.autograd.set_detect_anomaly(True)

    logits1 = net1(inten)
    logits1 = F.interpolate(logits1, inten.size()[2:], mode='bilinear', align_corners=True)
    logits2 = net2(inten)
    logits2 = F.interpolate(logits2, inten.size()[2:], mode='bilinear', align_corners=True)

    loss1 = criteria1(logits1, lbs)
    loss2 = criteria2(logits2, lbs.clone())
    loss = loss1 + loss2
    loss.backward()


