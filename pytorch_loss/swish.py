#!/usr/bin/python
# -*- encoding: utf-8 -*-


import torch
import torch.nn as nn
import torch.cuda.amp as amp


##
# version 1: use pytorch autograd
class SwishV1(nn.Module):

    def __init__(self):
        super(SwishV1, self).__init__()

    def forward(self, feat):
        return feat * torch.sigmoid(feat)


##
# version 2: use derived formula to compute grad
class SwishFunction(torch.autograd.Function):

    @staticmethod
    @amp.custom_fwd
    def forward(ctx, feat):
        sig = torch.sigmoid(feat)
        out = feat * torch.sigmoid(feat)
        grad = sig * (1 + feat * (1 - sig))
        ctx.grad = grad
        return out

    @staticmethod
    @amp.custom_bwd
    def backward(ctx, grad_output):
        grad = ctx.grad
        grad *= grad_output
        return grad


class SwishV2(nn.Module):

    def __init__(self):
        super(SwishV2, self).__init__()

    def forward(self, feat):
        return SwishFunction.apply(feat)


##
# version 3: write with cuda which requires less memory and can be faster
import swish_cpp
class SwishFunctionV3(torch.autograd.Function):

    @staticmethod
    @amp.custom_fwd
    def forward(ctx, feat):
        ctx.feat = feat
        return swish_cpp.swish_forward(feat)

    @staticmethod
    @amp.custom_bwd
    def backward(ctx, grad_output):
        feat = ctx.feat
        return swish_cpp.swish_backward(grad_output, feat)


class SwishV3(nn.Module):

    def __init__(self):
        super(SwishV3, self).__init__()

    def forward(self, feat):
        return SwishFunctionV3.apply(feat)


if __name__ == "__main__":
    import torchvision
    net = torchvision.models.resnet50(pretrained=True)
    sd = {k: v for k, v in net.state_dict().items() if k.startswith('conv1.') or k.startswith('bn1.')}

    class Net(nn.Module):
        def __init__(self, act='swishv1'):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(3, 64, 7, 2, 3)
            self.bn1 = nn.BatchNorm2d(64)
            if act == 'swishv1':
                self.act1 = SwishV1()
            elif act == 'swishv2':
                self.act1 = SwishV2()
            elif act == 'swishv3':
                self.act1 = SwishV3()
            self.dense = nn.Linear(64, 10, bias=False)
            self.crit = nn.CrossEntropyLoss()
            state = self.state_dict()
            state.update(sd)
            self.load_state_dict(state)
            torch.nn.init.constant_(self.dense.weight, 1)
        def forward(self, feat, label):
            feat = self.conv1(feat)
            feat = self.bn1(feat)
            feat = self.act1(feat)
            feat = torch.mean(feat, dim=(2, 3))
            logits = self.dense(feat)
            loss = self.crit(logits, label)
            return loss

    net1 = Net(act='swishv1')
    net2 = Net(act='swishv3')
    net2.load_state_dict(net1.state_dict())
    net1.cuda()
    net2.cuda()
    opt1 = torch.optim.SGD(net1.parameters(), lr=1e-3)
    opt2 = torch.optim.SGD(net2.parameters(), lr=1e-3)
    bs = 32
    for i in range(10000):
        inten = torch.randn(bs, 3, 224, 224).cuda().detach()
        label = torch.randint(0, 10, (bs, )).cuda().detach()

        loss1 = net1(inten, label)
        opt1.zero_grad()
        loss1.backward()
        opt1.step()

        loss2 = net2(inten, label)
        opt2.zero_grad()
        loss2.backward()
        opt2.step()

        if i % 200 == 0:
            print('====')
            print('loss diff: ', loss1.item() - loss2.item())
            print('weight diff: ', torch.sum(torch.abs(net1.conv1.weight - net2.conv1.weight)).item())

    from torch.autograd import gradcheck
    inten = torch.randn(3, 4, 6, 6).cuda()
    inten.requires_grad_(True)
    gradcheck(SwishFunctionV3.apply, [inten, ])



