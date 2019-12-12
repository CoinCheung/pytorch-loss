#!/usr/bin/python
# -*- encoding: utf-8 -*-


import torch
import torch.nn as nn


class LabelSmoothSoftmaxCEV1(nn.Module):

    def __init__(self, lb_smooth=0.1, reduction='mean', lb_ignore=-100):
        super(LabelSmoothSoftmaxCEV1, self).__init__()
        self.lb_smooth = lb_smooth
        self.reduction = reduction
        self.lb_ignore = lb_ignore
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, logits, label):
        '''
        args: logits: tensor of shape (N, C, H, W)
        args: label: tensor of shape(N, H, W)
        '''
        # overcome ignored label
        with torch.no_grad():
            num_classes = logits.size(1)
            label = label.clone().detach()
            ignore = label == self.lb_ignore
            n_valid = (ignore == 0).sum()
            label[ignore] = 0
            lb_pos, lb_neg = 1. - self.lb_smooth, self.lb_smooth / num_classes
            label = torch.empty_like(logits).fill_(
                lb_neg).scatter_(1, label.unsqueeze(1), lb_pos).detach()

        logs = self.log_softmax(logits)
        loss = -torch.sum(logs * label, dim=1)
        loss[ignore] = 0
        if self.reduction == 'mean':
            loss = loss.sum() / n_valid
        if self.reduction == 'sum':
            loss = loss.sum()

        return loss



class LabelSmoothSoftmaxCEFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, logits, label, lb_smooth, reduction, lb_ignore):
        # prepare label
        num_classes = logits.size(1)
        label = label.clone().detach()
        ignore = label == lb_ignore
        n_valid = (ignore == 0).sum()
        label[ignore] = 0
        lb_pos, lb_neg = 1. - lb_smooth, lb_smooth / num_classes
        label = torch.empty_like(logits).fill_(
            lb_neg).scatter_(1, label.unsqueeze(1), lb_pos).detach()

        scores = torch.softmax(logits, dim=1)
        logs = torch.log(scores)

        ignore = ignore.nonzero()
        _, M = ignore.size()
        a, *b = ignore.chunk(M, dim=1)
        scores[[a, torch.arange(scores.size(1)), *b]] = 0
        label[[a, torch.arange(label.size(1)), *b]] = 0

        ctx.scores = scores
        ctx.label = label
        ctx.reduction = reduction
        ctx.n_valid = n_valid

        loss = -torch.sum(logs * label, dim=1)
        if reduction == 'mean':
            loss = loss.sum() / n_valid
        if reduction == 'sum':
            loss = loss.sum()
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        scores = ctx.scores
        label = ctx.label
        reduction = ctx.reduction
        n_valid = ctx.n_valid
        if reduction == 'none':
            grad = grad_output.unsqueeze(1) * (scores - label)
        elif reduction == 'sum':
            grad = grad_output * (scores - label)
        elif reduction == 'mean':
            grad_output /= n_valid
            grad = grad_output * (scores - label)
        return grad, None, None, None, None, None


class LabelSmoothSoftmaxCEV2(nn.Module):

    def __init__(self, lb_smooth=0.1, reduction='mean', lb_ignore=-100):
        super(LabelSmoothSoftmaxCEV2, self).__init__()
        self.lb_smooth = lb_smooth
        self.reduction = reduction
        self.lb_ignore = lb_ignore

    def forward(self, logits, label):
        return LabelSmoothSoftmaxCEFunction.apply(
                logits, label, self.lb_smooth, self.reduction, self.lb_ignore)




if __name__ == '__main__':
    import torchvision
    import torch
    import numpy as np
    import random
    torch.manual_seed(15)
    random.seed(15)
    np.random.seed(15)
    torch.backends.cudnn.deterministic = True
    net1 = torchvision.models.resnet18(pretrained=True)
    net2 = torchvision.models.resnet18(pretrained=True)
    criteria1 = LabelSmoothSoftmaxCEV1(lb_smooth=0.1, lb_ignore=255)
    criteria2 = LabelSmoothSoftmaxCEV2(lb_smooth=0.1, lb_ignore=255)
    net1.cuda()
    net2.cuda()
    net1.train()
    net2.train()
    criteria1.cuda()
    criteria2.cuda()

    optim1 = torch.optim.SGD(net1.parameters(), lr=1e-2)
    optim2 = torch.optim.SGD(net2.parameters(), lr=1e-2)

    bs = 128
    for it in range(300000):
        inten = torch.randn(bs, 3, 224, 244).cuda()
        inten[0, 1, 0, 0] = 255
        inten[0, 0, 1, 2] = 255
        inten[0, 2, 5, 28] = 255
        lbs = torch.randint(0, 1000, (bs, )).cuda()
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
        #  print(net2.fc.weight[:, :5])
        with torch.no_grad():
            if (it+1) % 50 == 0:
                print('iter: {}, ================='.format(it+1))
                #  print(net1.fc.weight.numel())
                print(torch.mean(torch.abs(net1.fc.weight - net2.fc.weight)).item())
                print(torch.mean(torch.abs(net1.conv1.weight - net2.conv1.weight)).item())
                #  print(loss1.item())
                #  print(loss2.item())
                print(loss1.item() - loss2.item())
