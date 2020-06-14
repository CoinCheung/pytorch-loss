#!/usr/bin/python
# -*- encoding: utf-8 -*-


import torch
import torch.nn as nn



##
# version 1: use torch.autograd
class LabelSmoothSoftmaxCEV1(nn.Module):
    '''
    This is the autograd version, you can also try the LabelSmoothSoftmaxCEV2 that uses derived gradients
    '''

    def __init__(self, lb_smooth=0.1, reduction='mean', ignore_index=-100):
        super(LabelSmoothSoftmaxCEV1, self).__init__()
        self.lb_smooth = lb_smooth
        self.reduction = reduction
        self.lb_ignore = ignore_index
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, logits, label):
        '''
        args: logits: tensor of shape (N, C, H, W)
        args: label: tensor of shape(N, H, W)
        '''
        # overcome ignored label
        logits = logits.float() # use fp32 to avoid nan
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



##
# version 2: user derived grad computation
class LSRCrossEntropyFunctionV2(torch.autograd.Function):

    @staticmethod
    def forward(ctx, logits, label, lb_smooth, lb_ignore):
        # prepare label
        num_classes = logits.size(1)
        lb_pos, lb_neg = 1. - lb_smooth, lb_smooth / num_classes
        label = label.clone().detach()
        ignore = label == lb_ignore
        n_valid = (label != lb_ignore).sum()
        label[ignore] = 0
        label = torch.empty_like(logits).fill_(
            lb_neg).scatter_(1, label.unsqueeze(1), lb_pos).detach()

        ignore = ignore.nonzero()
        _, M = ignore.size()
        a, *b = ignore.chunk(M, dim=1)
        mask = [a, torch.arange(label.size(1)), *b]
        label[mask] = 0
        coeff = (num_classes - 1) * lb_neg + lb_pos

        ctx.variables = coeff, mask, logits, label

        loss = torch.log_softmax(logits, dim=1).neg_().mul_(label).sum(dim=1)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        coeff, mask, logits, label = ctx.variables

        scores = torch.softmax(logits, dim=1).mul_(coeff)
        grad = scores.sub_(label).mul_(grad_output.unsqueeze(1))
        grad[mask] = 0
        return grad, None, None, None


class LabelSmoothSoftmaxCEV2(nn.Module):

    def __init__(self, lb_smooth=0.1, reduction='mean', ignore_index=-100):
        super(LabelSmoothSoftmaxCEV2, self).__init__()
        self.lb_smooth = lb_smooth
        self.reduction = reduction
        self.lb_ignore = ignore_index

    def forward(self, logits, labels):
        losses = LSRCrossEntropyFunctionV2.apply(
                logits, labels, self.lb_smooth, self.lb_ignore)
        if self.reduction == 'sum':
            losses = losses.sum()
        elif self.reduction == 'mean':
            n_valid = (labels != self.lb_ignore).sum()
            losses = losses.sum() / n_valid
        return losses

##
# version 3: implement wit cpp/cuda to save memory and accelerate
import lsr_cpp
class LSRCrossEntropyFunctionV3(torch.autograd.Function):
    '''
    use cpp/cuda to accelerate and shrink memory usage
    '''
    @staticmethod
    def forward(ctx, logits, labels, lb_smooth, lb_ignore):
        losses = lsr_cpp.lsr_forward(logits, labels, lb_ignore, lb_smooth)

        ctx.variables = logits, labels, lb_ignore, lb_smooth
        return losses

    @staticmethod
    def backward(ctx, grad_output):
        logits, labels, lb_ignore, lb_smooth = ctx.variables

        grad = lsr_cpp.lsr_backward(grad_output, logits, labels, lb_ignore, lb_smooth)
        return grad, None, None, None


class LabelSmoothSoftmaxCEV3(nn.Module):

    def __init__(self, lb_smooth=0.1, reduction='mean', ignore_index=-100):
        super(LabelSmoothSoftmaxCEV3, self).__init__()
        self.lb_smooth = lb_smooth
        self.reduction = reduction
        self.lb_ignore = ignore_index

    def forward(self, logits, labels):
        losses = LSRCrossEntropyFunctionV3.apply(
                logits, labels, self.lb_smooth, self.lb_ignore)
        if self.reduction == 'sum':
            losses = losses.sum()
        elif self.reduction == 'mean':
            n_valid = (labels != self.lb_ignore).sum()
            losses = losses.sum() / n_valid
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
    net1 = torchvision.models.resnet18(pretrained=True)
    net2 = torchvision.models.resnet18(pretrained=True)
    net2.load_state_dict(net1.state_dict())
    red = 'mean'
    criteria1 = LabelSmoothSoftmaxCEV3(lb_smooth=0.1, ignore_index=255, reduction=red)
    criteria2 = LabelSmoothSoftmaxCEV2(lb_smooth=0.1, ignore_index=255, reduction=red)
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
        lbs = torch.randint(0, 1000, (bs, )).cuda()
        lbs[1] = 255
        lbs[30] = 255
        lbs[108] = 255
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
        #  net2.load_state_dict(net1.state_dict())
        #  print(net2.fc.weight[:, :5])
        with torch.no_grad():
            if (it+1) % 50 == 0:
                print('iter: {}, ================='.format(it+1))
                #  print(net1.fc.weight.numel())
                print('fc weight: ', torch.mean(torch.abs(net1.fc.weight - net2.fc.weight)).item())

                print('conv1 weight: ', torch.mean(torch.abs(net1.conv1.weight - net2.conv1.weight)).item())
                #  print(loss1.item())
                #  print(loss2.item())
                print('loss: ', loss1.item() - loss2.item())
