#!/usr/bin/python
# -*- encoding: utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F


## use autograd
class MishV1(nn.Module):

    def __init__(self):
        super(MishV1, self).__init__()

    def forward(self, feat):
        return feat * torch.tanh(F.softplus(feat))


## use self-computed back-propagation, maybe use less memory and faster
class MishFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, feat):
        #  exp = torch.exp(feat)
        #  exp_plus = exp + 1
        #  exp_plus_pow = torch.pow(exp_plus, 2)
        #  tanhX = (exp_plus_pow - 1) / (exp_plus_pow + 1)
        #  out = feat * tanhX
        #  grad = tanhX + 4 * feat * exp * exp_plus / torch.pow(1 + exp_plus_pow, 2)

        tanhX = torch.tanh(F.softplus(feat))
        out = feat * tanhX
        grad = tanhX + feat * (1 - torch.pow(tanhX, 2)) * torch.sigmoid(feat)

        ctx.grad = grad
        return out

    @staticmethod
    def backward(ctx, grad_output):
        grad = ctx.grad
        grad *= grad_output
        return grad


class MishV2(nn.Module):

    def __init__(self):
        super(MishV2, self).__init__()

    def forward(self, feat):
        return MishFunction.apply(feat)


if __name__ == "__main__":
    import torchvision
    net = torchvision.models.resnet50(pretrained=True)
    sd = {k: v for k, v in net.state_dict().items() if k.startswith('conv1.') or k.startswith('bn1.')}

    class Net(nn.Module):
        def __init__(self, act='mishv1'):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(3, 64, 7, 2, 3)
            self.bn1 = nn.BatchNorm2d(64)
            if act == 'mishv1':
                self.act = MishV1()
            elif act == 'mishv2':
                self.act = MishV2()
            self.dense = nn.Linear(64, 10, bias=False)
            self.crit = nn.CrossEntropyLoss()
            #  state = self.state_dict()
            #  state.update(sd)
            #  self.load_state_dict(state)
            #  torch.nn.init.constant_(self.dense.weight, 1)
        def forward(self, feat, label):
            feat = self.conv1(feat)
            feat = self.bn1(feat)
            feat = self.act(feat)
            feat = torch.mean(feat, dim=(2, 3))
            logits = self.dense(feat)
            loss = self.crit(logits, label)
            return loss

    net1 = Net(act='mishv2')
    #  net2 = Net(act='mishv2')
    #  net2.load_state_dict(net1.state_dict())
    net1.cuda()
    #  net2.cuda()
    opt1 = torch.optim.SGD(net1.parameters(), lr=1e-1)
    #  opt2 = torch.optim.SGD(net2.parameters(), lr=1e-1)
    for i in range(2000):
        inten = torch.randn(16, 3, 512, 512).cuda().detach()
        label = torch.randint(0, 10, (16, )).cuda().detach()

        loss1 = net1(inten, label)
        opt1.zero_grad()
        loss1.backward()
        opt1.step()

        #  loss2 = net2(inten, label)
        #  opt2.zero_grad()
        #  loss2.backward()
        #  opt2.step()

        #  if i % 200 == 0:
        #      #  print('====')
        #      #  print(loss1.item() - loss2.item())
        #      print(torch.sum(torch.abs(net1.conv1.weight) - torch.abs(net2.conv1.weight)).item())
        #


