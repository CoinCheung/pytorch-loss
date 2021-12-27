#!/usr/bin/python
# -*- encoding: utf-8 -*-

'''
LayerNorm working in same way as pytorch native implementation, but usage is more similar to nn.BatchNorm. I write this for better support of visual transformers.
Sadly, the cuda implementation is just on-par with a combination of pytorch native operators in terms of speed and memory usage (speed is faster but not noticeable). Maybe LayerNorm is not the bottleneck of my model, and I should not waste my time writing cuda code here. pytorch native operators are good enough here.
The cuda kernel and V2 version has the problem of nan during backward pass in float16 mode(not always nan, only when input values are very big which causes sum(x**2) to be out of range of a fp16 number), thus I simply cast the input into float32 when input is float16.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.amp as amp
import layer_norm_cpp # should import torch before import this


# v1: pytorch autograd
class LayerNormV1(nn.Module):
    '''
    '''
    def __init__(self, n_chan, affine=True, eps=1e-6):
        super(LayerNormV1, self).__init__()
        self.n_chan, self.affine = n_chan, affine
        self.weight, self.bias = None, None
        self.eps = eps
        if affine:
            self.weight = nn.Parameter(torch.ones(1, n_chan, 1))
            self.bias = nn.Parameter(torch.zeros(1, n_chan, 1))

    def forward(self, x):
        '''
        input is NCHW, norm along C
        '''
        N, C, H, W = x.size()
        x = x.view(N, C, -1)
        mean = x.mean(dim=1, keepdim=True)
        std = (x.var(dim=1, keepdim=True, unbiased=False) + self.eps).rsqrt()
        x = (x - mean) * std
        if self.affine:
            x = self.weight * x + self.bias
        x = x.view(N, C, H, W)
        return x


##
# v2: self-derived grad formula
class LayerNormV2(nn.Module):
    '''
    '''
    def __init__(self, n_chan, affine=True, eps=1e-6):
        super(LayerNormV2, self).__init__()
        self.n_chan, self.affine = n_chan, affine
        self.weight, self.bias = None, None
        self.eps = eps
        if affine:
            self.weight = nn.Parameter(torch.ones(1, n_chan, 1))
            self.bias = nn.Parameter(torch.zeros(1, n_chan, 1))

    def forward(self, x):
        '''
        input is NCHW, norm along C
        '''
        N, C, H, W = x.size()
        x = x.view(N, C, -1)
        dt = x.dtype
        if dt == torch.float16: x = x.float()
        x = LayerNormV2Func.apply(x, self.eps).to(dt)
        if self.affine: x = self.weight * x + self.bias
        x = x.view(N, C, H, W)
        return x


class LayerNormV2Func(torch.autograd.Function):
    '''
    compute backward directly for better numeric stability
    '''
    @staticmethod
    @amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, x, eps):
        '''
        inputs:
            x: (N, C, M)
            eps: float
        outpus:
            x: (N, C, M)
        '''
        mean = x.mean(dim=1, keepdim=True)
        std = (x.var(dim=1, keepdim=True, unbiased=False) + eps).rsqrt()
        out = (x - mean).mul_(std)
        ctx.vars = x, eps
        return out

    @staticmethod
    @amp.custom_bwd
    def backward(ctx, grad_output):
        '''
        '''
        x, eps = ctx.vars
        N, C, M = x.size()
        mean = x.mean(dim=1, keepdim=True)
        var_plus_eps = x.var(dim=1, keepdim=True, unbiased=False) + eps

        grads = (x - mean).mul_(x - 1/C).mul_(x.sum(dim=1, keepdim=True)).mul_(var_plus_eps).add_(1).mul_(var_plus_eps.rsqrt()).mul_(1./C).mul_(grad_output)

        return grads, None


##
# v3: implement with cuda to save memory
class LayerNormV3(nn.Module):
    '''
    '''
    def __init__(self, n_chan, affine=True, eps=1e-6):
        super(LayerNormV3, self).__init__()
        self.n_chan, self.affine = n_chan, affine
        self.weight, self.bias = None, None
        self.eps = eps
        if affine:
            self.weight = nn.Parameter(torch.ones(1, n_chan, 1))
            self.bias = nn.Parameter(torch.zeros(1, n_chan, 1))

    def forward(self, x):
        '''
        input is NCHW, norm along C
        '''
        N, C, H, W = x.size()
        x = x.view(N, C, -1)
        dt = x.dtype
        if dt == torch.float16: x = x.float()
        x = LayerNormV3Func.apply(x, self.eps).to(dt)
        if self.affine: x = self.weight * x + self.bias
        x = x.view(N, C, H, W)
        return x


class LayerNormV3Func(torch.autograd.Function):
    '''
    compute backward directly for better numeric stability
    '''
    @staticmethod
    @amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, x, eps):
        '''
        inputs:
            x: (N, C, M)
            eps: float
        outpus:
            x: (N, C, M)
        '''
        out = layer_norm_cpp.layer_norm_forward(x, eps)
        ctx.vars = x, eps
        return out

    @staticmethod
    @amp.custom_bwd
    def backward(ctx, grad_output):
        '''
        '''
        x, eps = ctx.vars
        grads = layer_norm_cpp.layer_norm_backward(grad_output, x, eps)
        return grads, None



if __name__ == '__main__':
    import torchvision
    import torch
    import numpy as np
    import random
    #  torch.manual_seed(15)
    #  random.seed(15)
    #  np.random.seed(15)
    #  torch.backends.cudnn.deterministic = True
    #  torch.cuda.set_device('cuda:1')

    class Model(nn.Module):
        def __init__(self, norm):
            super(Model, self).__init__()
            net = torchvision.models.resnet18(pretrained=False)
            self.conv1 = net.conv1
            self.bn1 = net.bn1
            self.maxpool = net.maxpool
            self.relu = net.relu
            self.layer1 = net.layer1
            self.layer2 = net.layer2
            self.layer3 = net.layer3
            self.layer4 = net.layer4
            self.out = nn.Conv2d(512, 1, 3, 1, 1)
            self.bn1 = norm(64)
            affine = True
            eps = 1e-6
            self.layer1[1].bn1 = norm(64, affine=affine, eps=eps)
            self.layer2[0].bn1 = norm(128, affine=affine, eps=eps)
            self.layer2[1].bn1 = norm(128, affine=affine, eps=eps)
            self.layer3[1].bn2 = norm(256, affine=affine, eps=eps)
            self.layer4[0].bn2 = norm(512, affine=affine, eps=eps)
            self.layer4[1].bn2 = norm(512, affine=affine, eps=eps)
        def forward(self, x):
            feat = self.conv1(x)
            feat = self.bn1(feat)
            feat = self.relu(feat)
            feat = self.maxpool(feat)
            feat = self.layer1(feat)
            feat = self.layer2(feat)
            feat = self.layer3(feat)
            feat = self.layer4(feat)
            feat = self.out(feat)
            out = F.interpolate(feat, x.size()[2:], mode='bilinear', align_corners=True)
            return out
    net1 = Model(norm=LayerNormV1)
    net2 = Model(norm=LayerNormV3)
    net2.load_state_dict(net1.state_dict())
    #  print(net1)

    criteria1 = nn.CrossEntropyLoss()
    criteria2 = nn.CrossEntropyLoss()
    net1.cuda()
    net2.cuda()
    net1.train()
    net2.train()
    net1.double()
    net2.double()
    #  net1.half()
    #  net2.half()
    criteria1.cuda()
    criteria2.cuda()

    optim1 = torch.optim.SGD(net1.parameters(), lr=1e-2)
    optim2 = torch.optim.SGD(net2.parameters(), lr=1e-2)

    norm1 = LayerNormV1(256)
    norm3 = LayerNormV3(256)
    norm1.cuda()
    norm3.cuda()
    #  inten = torch.randn(2, 256, 80, 80).cuda().half()
    inten = torch.randn(2, 256, 80, 80).cuda().double()
    #  inten = torch.randn(2, 256, 80, 80).cuda()
    out1 = norm1(inten)
    out3 = norm3(inten)
    print('diff: ', torch.abs((out1 - out3)).max())
    #  print('inten.mean(): ', inten.mean(dim=1))
    #  print('inten.var(): ', inten.var(dim=1, unbiased=False))
    #  print('inten.pow(2).sum(): ', inten.pow(2).sum(dim=1))

    bs = 12
    size = 640, 640
    #  size = 229, 229
    for it in range(10):
        #  inten = torch.randn(bs, 3, *size).cuda().half()
        inten = torch.randn(bs, 3, *size).cuda().double()
        #  inten = torch.randn(bs, 3, *size).cuda()
        inten[0][0][0] = 444.
        lbs = torch.randint(0, 1, (bs, *size)).cuda()
        #  inten = inten.double()
        lbs = lbs

        logits1 = net1(inten)
        loss1 = criteria1(logits1, lbs)
        optim1.zero_grad()
        loss1.backward()
        optim1.step()
        logits2 = net2(inten)
        loss2 = criteria2(logits2, lbs)
        optim2.zero_grad()
        loss2.backward()
        optim2.step()

        #  print(logits1.isnan().sum())
        #  print(logits2.isnan().sum())
        #  print((1-logits2).bool().isnan().sum())
        #  print(logits2.numel())
        #  print('diff: ', torch.abs((logits1 - logits2)).max())
        print('diff: ', torch.abs((logits1 - logits2)).max())

        #  with torch.no_grad():
        #      if (it+1) % 50 == 0:
        #          print('iter: {}, ================='.format(it+1))
        #          print('out.weight: ', torch.mean(torch.abs(net1.out.weight - net2.out.weight)).item())
        #          print('conv1.weight: ', torch.mean(torch.abs(net1.conv1.weight - net2.conv1.weight)).item())
        #          print('loss: ', loss1.item() - loss2.item())
