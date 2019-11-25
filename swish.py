#!/usr/bin/python
# -*- encoding: utf-8 -*-


import torch
import torch.nn as nn


## use autograd
class SwishV1(nn.Module):

    def __init__(self):
        super(SwishV1, self).__init__()

    def forward(self, feat):
        return feat * torch.sigmoid(feat)


## use self-computed back-propagation, use less memory and faster
class SwishFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, feat):
        sig = torch.sigmoid(feat)
        out = feat * torch.sigmoid(feat)
        grad = sig * (1 + feat * (1 - sig))
        ctx.grad = grad
        return out

    @staticmethod
    def backward(ctx, grad_output):
        grad = ctx.grad
        grad *= grad_output
        return grad


class SwishV2(nn.Module):

    def __init__(self):
        super(SwishV2, self).__init__()

    def forward(self, feat):
        return SwishFunction.apply(feat)


if __name__ == "__main__":
    import torchvision
    net = torchvision.models.resnet50(pretrained=True)
    sd = {k: v for k, v in net.state_dict().items() if k.startswith('conv1.') or k.startswith('bn1.')}
    print(sd)

    class Net(nn.Module):
        def __init__(self, act='swishv1'):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(3, 64, 7, 2, 3)
            self.bn1 = nn.BatchNorm2d(64)
            if act == 'swishv1':
                self.act = SwishV1()
            else:
                self.act = SwishV2()
            self.dense = nn.Linear(64, 10, bias=False)
            self.crit = nn.CrossEntropyLoss()
            state = self.state_dict()
            state.update(sd)
            self.load_state_dict(state)
            torch.nn.init.constant_(self.dense.weight, 1)
        def forward(self, feat, label):
            feat = self.conv1(feat)
            feat = self.bn1(feat)
            feat = self.act(feat)
            feat = torch.mean(feat, dim=(2, 3))
            logits = self.dense(feat)
            loss = self.crit(logits, label)
            return loss

    net1 = Net(act='swishv1')
    net2 = Net(act='swishv2')
    opt1 = torch.optim.SGD(net1.parameters(), lr=1e-3)
    opt2 = torch.optim.SGD(net2.parameters(), lr=1e-3)
    for i in range(10):
        inten = torch.randn(16, 3, 512, 512).detach()
        label = torch.randint(0, 10, (16, )).detach()

        loss1 = net1(inten, label)
        opt1.zero_grad()
        loss1.backward()
        opt1.step()

        loss2 = net2(inten, label)
        opt2.zero_grad()
        loss2.backward()
        opt2.step()

        print(loss1.item() - loss2.item())



