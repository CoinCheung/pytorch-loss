#!/usr/bin/python
# -*- encoding: utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F
import  torch.cuda.amp as amp

'''
    Proposed in this paper:  https://arxiv.org/abs/1911.10688
'''


def pc_softmax_func(logits, lb_proportion):
    assert logits.size(1) == len(lb_proportion)
    shape = [1, -1] + [1 for _ in range(len(logits.size()) - 2)]
    W = torch.tensor(lb_proportion).view(*shape).to(logits.device).detach()
    logits = logits - logits.max(dim=1, keepdim=True)[0]
    exp = torch.exp(logits)
    pc_softmax = exp.div_((W * exp).sum(dim=1, keepdim=True))
    return pc_softmax


class PCSoftmax(nn.Module):

    def __init__(self, lb_proportion):
        super(PCSoftmax, self).__init__()
        self.weight = lb_proportion

    def forward(self, logits):
        return pc_softmax_func(logits, self.weight)


class PCSoftmaxCrossEntropyV1(nn.Module):

    def __init__(self, lb_proportion, ignore_index=255, reduction='mean'):
        super(PCSoftmaxCrossEntropyV1, self).__init__()
        self.weight = torch.tensor(lb_proportion).cuda().detach()
        self.nll = nn.NLLLoss(reduction=reduction, ignore_index=ignore_index)

    def forward(self, logits, label):
        shape = [1, -1] + [1 for _ in range(len(logits.size()) - 2)]
        W = self.weight.view(*shape).to(logits.device).detach()
        logits = logits - logits.max(dim=1, keepdim=True)[0]
        wexp_sum = torch.exp(logits).mul(W).sum(dim=1, keepdim=True)
        log_wsoftmax = logits - torch.log(wexp_sum)
        loss = self.nll(log_wsoftmax, label)
        return loss


class PCSoftmaxCrossEntropyFunction(torch.autograd.Function):

    @staticmethod
    @amp.custom_fwd
    def forward(ctx, logits, label, lb_proportion, reduction, ignore_index):
        # prepare label
        label = label.clone().detach()
        ignore = label == ignore_index
        n_valid = (ignore == 0).sum()
        label[ignore] = 0
        lb_one_hot = torch.zeros_like(logits).scatter_(
            1, label.unsqueeze(1), 1).detach()

        shape = [1, -1] + [1 for _ in range(len(logits.size()) - 2)]
        W = torch.tensor(lb_proportion).view(*shape).to(logits.device).detach()
        logits = logits - logits.max(dim=1, keepdim=True)[0]
        exp_wsum = torch.exp(logits).mul_(W).sum(dim=1, keepdim=True)

        ignore = ignore.nonzero()
        _, M = ignore.size()
        a, *b = ignore.chunk(M, dim=1)
        mask = [a, torch.arange(lb_one_hot.size(1)), *b]
        lb_one_hot[mask] = 0

        ctx.mask = mask
        ctx.W = W
        ctx.lb_one_hot = lb_one_hot
        ctx.logits = logits
        ctx.exp_wsum = exp_wsum
        ctx.reduction = reduction
        ctx.n_valid = n_valid

        log_wsoftmax = logits - torch.log(exp_wsum)
        loss = -log_wsoftmax.mul_(lb_one_hot).sum(dim=1)
        if reduction == 'mean':
            loss = loss.sum().div_(n_valid)
        if reduction == 'sum':
            loss = loss.sum()
        return loss

    @staticmethod
    @amp.custom_bwd
    def backward(ctx, grad_output):
        mask = ctx.mask
        W = ctx.W
        lb_one_hot = ctx.lb_one_hot
        logits = ctx.logits
        exp_wsum = ctx.exp_wsum
        reduction = ctx.reduction
        n_valid = ctx.n_valid

        wlabel = torch.sum(W * lb_one_hot, dim=1, keepdim=True)
        wscores = torch.exp(logits).div_(exp_wsum).mul_(wlabel)
        wscores[mask] = 0
        grad = wscores.sub_(lb_one_hot)

        if reduction == 'none':
            grad.mul_(grad_output.unsqueeze(1))
        elif reduction == 'sum':
            grad.mul_(grad_output)
        elif reduction == 'mean':
            grad.div_(n_valid).mul_(grad_output)
        return grad, None, None, None, None, None


class PCSoftmaxCrossEntropyV2(nn.Module):

    def __init__(self, lb_proportion, reduction='mean', ignore_index=-100):
        super(PCSoftmaxCrossEntropyV2, self).__init__()
        self.lb_proportion = lb_proportion
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, logits, label):
        return PCSoftmaxCrossEntropyFunction.apply(
            logits, label, self.lb_proportion, self.reduction, self.ignore_index)



if __name__ == "__main__":

    torch.backends.cudnn.deterministic = True
    import torchvision
    net1 = torchvision.models.resnet18()
    net1.fc = nn.Linear(512, 19)
    net1.cuda()
    net2 = torchvision.models.resnet18()
    net2.fc = nn.Linear(512, 19)
    net2.cuda()
    net2.load_state_dict(net1.state_dict())

    lb_proportion = [1. for _ in range(19)]
    crit1 = nn.CrossEntropyLoss()
    #  crit2 = nn.CrossEntropyLoss()
    crit1 = PCSoftmaxCrossEntropyV1(lb_proportion)
    crit2 = PCSoftmaxCrossEntropyV2(lb_proportion)
    optim1 = torch.optim.SGD(net1.parameters(), lr=1e-3)
    optim2 = torch.optim.SGD(net2.parameters(), lr=1e-3)

    for i in range(1000):
        inten = torch.randn(8, 3, 224, 224).cuda()
        lb = torch.randint(0, 19, (8,)).cuda()
        logits1 = net1(inten)
        logits2 = net2(inten)

        #  logits = torch.randn(8, 19, 224, 224).cuda()
        #  lb = torch.randint(0, 19, (8, 224, 224)).cuda()
        #  logits = torch.tensor(logits, requires_grad=True)

        loss1 = crit1(logits1, lb)
        loss2 = crit2(logits2, lb)
        optim1.zero_grad()
        optim2.zero_grad()
        loss1.backward()
        loss2.backward()
        optim1.step()
        optim2.step()
        #  print(loss1.item())
        if i % 100 == 0:
            #  print(loss2.item() - loss1.item())
            print((net1.conv1.weight - net2.conv1.weight).abs().max().item())
            #  print((net1.fc.weight - net2.fc.weight).abs().max().item())



    #  lb_proportion = [1. for _ in range(3)]
    #  diff = torch.softmax(inten, dim=1) - pc_softmax_func(inten, lb_proportion)
    #  print(torch.max(diff))
    #  print(torch.min(diff))

    #  loss1.backward()
