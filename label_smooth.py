#!/usr/bin/python
# -*- encoding: utf-8 -*-


import torch
import torch.nn as nn


class LabelSmoothSoftmaxCE(nn.Module):
    def __init__(self,
                 lb_pos=0.9,
                 lb_neg=0.005,
                 reduction='mean',
                 lb_ignore=255,
                 ):
        super(LabelSmoothSoftmaxCE, self).__init__()
        self.lb_pos = lb_pos
        self.lb_neg = lb_neg
        self.reduction = reduction
        self.lb_ignore = lb_ignore
        self.log_softmax = nn.LogSoftmax(1)

    def forward(self, logits, label):
        logs = self.log_softmax(logits)
        ignore = label.data.cpu() == self.lb_ignore
        n_valid = (ignore == 0).sum()
        label = label.clone()
        label[ignore] = 0
        lb_one_hot = logits.data.clone().zero_().scatter_(1, label.unsqueeze(1), 1)
        label = self.lb_pos * lb_one_hot + self.lb_neg * (1-lb_one_hot)
        label = label.detach()

        loss = -torch.sum(logs*label, dim=1)
        loss[ignore] = 0
        if self.reduction == 'mean':
            loss = loss.sum() / n_valid
        elif self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'none':
            loss = loss
        return loss


if __name__ == '__main__':
    torch.manual_seed(15)
    criteria = LabelSmoothSoftmaxCE(lb_pos=0.9, lb_neg=5e-3)
    net1 = nn.Sequential(
        nn.Conv2d(3, 3, kernel_size=3, stride=2, padding=1),
    )
    net1.cuda()
    net1.train()
    net2 = nn.Sequential(
        nn.Conv2d(3, 3, kernel_size=3, stride=2, padding=1),
    )
    net2.cuda()
    net2.train()

    with torch.no_grad():
        inten = torch.randn(2, 3, 5, 5).cuda()
        lbs = torch.randint(0, 3, [2, 5, 5]).cuda()
        lbs[1, 3, 4] = 255
        lbs[1, 2, 3] = 255
        print(lbs)

    import torch.nn.functional as F
    logits1 = net1(inten)
    logits1 = F.interpolate(logits1, inten.size()[2:], mode='bilinear', align_corners=True)
    logits2 = net2(inten)
    logits2 = F.interpolate(logits2, inten.size()[2:], mode='bilinear', align_corners=True)

    #  loss1 = criteria1(logits1, lbs)
    loss = criteria(logits1, lbs)
    #  print(loss.detach().cpu())
    loss.backward()
    print(loss)

