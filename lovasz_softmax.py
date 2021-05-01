#!/usr/bin/python
# -*- encoding: utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.amp as amp

#  grads = {}

##
# version 1: use torch.autograd
class LovaszSoftmaxV1(nn.Module):
    '''
    This is the autograd version, used in the multi-category classification case
    '''
    def __init__(self, reduction='mean', ignore_index=-100):
        super(LovaszSoftmaxV1, self).__init__()
        self.reduction = reduction
        self.lb_ignore = ignore_index

    def forward(self, logits, label):
        '''
        Same usage method as nn.CrossEntropyLoss:
            >>> criteria = LovaszSoftmaxV1()
            >>> logits = torch.randn(8, 19, 384, 384) # nchw, float/half
            >>> lbs = torch.randint(0, 19, (8, 384, 384)) # nhw, int64_t
            >>> loss = criteria(logits, lbs)
        '''
        # overcome ignored label
        n, c, h, w = logits.size()
        logits = logits.transpose(0, 1).reshape(c, -1).float() # use fp32 to avoid nan
        label = label.view(-1)

        idx = label.ne(self.lb_ignore).nonzero(as_tuple=False).squeeze()
        probs = logits.softmax(dim=0)[:, idx]

        label = label[idx]
        lb_one_hot = torch.zeros_like(probs).scatter_(
                0, label.unsqueeze(0), 1).detach()

        errs = (lb_one_hot - probs).abs()
        errs_sort, errs_order = torch.sort(errs, dim=1, descending=True)
        n_samples = errs.size(1)

        # lovasz extension grad
        with torch.no_grad():
            #  lb_one_hot_sort = lb_one_hot[
            #      torch.arange(c).unsqueeze(1).repeat(1, n_samples), errs_order
            #      ].detach()
            lb_one_hot_sort = torch.cat([
                lb_one_hot[i, ord].unsqueeze(0)
                for i, ord in enumerate(errs_order)], dim=0)
            n_pos = lb_one_hot_sort.sum(dim=1, keepdim=True)
            inter = n_pos - lb_one_hot_sort.cumsum(dim=1)
            union = n_pos + (1. - lb_one_hot_sort).cumsum(dim=1)
            jacc = 1. - inter / union
            if n_samples > 1:
                jacc[:, 1:] = jacc[:, 1:] - jacc[:, :-1]

        losses = torch.einsum('ab,ab->a', errs_sort, jacc)

        if self.reduction == 'sum':
            losses = losses.sum()
        elif self.reduction == 'mean':
            losses = losses.mean()
        return losses, errs



##
# version 3: use cuda
import lovasz_softmax_cpp
class LovaszSoftmaxFunctionV3(torch.autograd.Function):

    @staticmethod
    @amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, logits, labels, ignore_index):
        losses, jacc = lovasz_softmax_cpp.lovasz_softmax_forward(logits,
                labels, ignore_index)
        ctx.vars = logits, labels, jacc, ignore_index
        #  grads['one_hot'] = jacc
        return losses

    @staticmethod
    @amp.custom_bwd
    def backward(ctx, grad_output):
        logits, labels, jacc, ignore_index = ctx.vars
        grad = lovasz_softmax_cpp.lovasz_softmax_backward(grad_output, logits, labels, jacc, ignore_index)
        return grad, None, None


class LovaszSoftmaxV3(nn.Module):
    '''
    '''
    def __init__(self, reduction='mean', ignore_index=-100):
        super(LovaszSoftmaxV3, self).__init__()
        self.reduction = reduction
        self.lb_ignore = ignore_index

    def forward(self, logits, label):
        '''
        Same usage method as nn.CrossEntropyLoss:
            >>> criteria = LovaszSoftmaxV3()
            >>> logits = torch.randn(8, 19, 384, 384) # nchw, float/half
            >>> lbs = torch.randint(0, 19, (8, 384, 384)) # nhw, int64_t
            >>> loss = criteria(logits, lbs)
        '''
        # overcome ignored label
        losses = LovaszSoftmaxFunctionV3.apply(logits, label, self.lb_ignore)
        if self.reduction == 'sum':
            losses = losses.sum()
        elif self.reduction == 'mean':
            losses = losses.mean()
        return losses




if __name__ == '__main__':
    torch.manual_seed(123)
    torch.cuda.manual_seed(123)
    #  crit1 = LovaszSoftmaxV1(reduction='none', ignore_index=255)
    #  crit2 = lovasz_softmax_cpp.lovasz_softmax_forward
    #
    #  bs, c, h, w = 2, 19, 1000, 1000
    #  #  bs, c, h, w = 2, 18, 1240, 1240
    #  inten = torch.randn(bs, c, h, w).cuda()
    #  #  inten2 = inten1.clone()
    #  label = torch.randint(0, c, (bs, h, w)).cuda()
    #  #  label[0, :, :] = 255
    #  #  label[1, 13:20, 6] = 255
    #
    #  loss1, errs1, jacc1 = crit1(inten, label)
    #  loss2, jacc2 = crit2(inten, label, 255)
    #  print(loss1.size())
    #  print(loss2.size())
    #  print((loss1.view(-1) - loss2.view(-1)).abs().sum())
    #  print((jacc1.view(-1) - jacc2.view(-1)).abs().sum())
    #  print(loss1)
    #  print(loss2)



    #  print((jac1 - jac2).sum())
    #  print(jac1[1, :8])
    #  print(jac2[1, :8])

    import torchvision
    import torch
    import numpy as np
    import random
    torch.manual_seed(15)
    random.seed(15)
    np.random.seed(15)
    torch.backends.cudnn.deterministic = True

    scaler = amp.GradScaler()

    class Model(nn.Module):
        def __init__(self, n_classes):
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
            self.fc = nn.Conv2d(512, n_classes, 3, 1, 1)
        def forward(self, x):
            feat = self.conv1(x)
            feat = self.bn1(feat)
            feat = self.relu(feat)
            feat = self.maxpool(feat)
            feat = self.layer1(feat)
            feat = self.layer2(feat)
            feat = self.layer3(feat)
            feat = self.layer4(feat)
            feat = self.fc(feat)
            out = F.interpolate(feat, x.size()[2:], mode='bilinear', align_corners=True)
            return out

    c = 227
    net1 = Model(c)
    net2 = Model(c)
    net2.load_state_dict(net1.state_dict())
    red = 'none'
    criteria1 = LovaszSoftmaxV1(reduction='sum', ignore_index=255)
    criteria2 = LovaszSoftmaxV3(reduction='sum', ignore_index=255)
    net1.cuda()
    net2.cuda()
    net1.train()
    net2.train()
    #  net1 = net1.half()
    #  net2 = net2.half()
    criteria1.cuda()
    criteria2.cuda()

    optim1 = torch.optim.SGD(net1.parameters(), lr=1e-2)
    optim2 = torch.optim.SGD(net2.parameters(), lr=1e-2)
    weight = torch.randn(c).softmax(dim=0).cuda().detach()

    bs, h, w = 2, 400, 400
    use_fp16 = False
    for it in range(1000):
        inten = torch.randn(bs, 3, h, w).cuda()#.half()
        lbs = torch.randint(0, c, (bs, h, w)).cuda()
        #  lbs2 = lbs.clone()
        #  lbs[1, 1, 1] = 255
        #  lbs[0, 3:100, 2:100] = 255
        #  lbs[1, 4:70, 28:200] = 255
        optim1.zero_grad()
        logits1 = net1(inten)
        #  logits1.retain_grad()
        loss1, one_hot = criteria1(logits1, lbs)
        loss1 = loss1.mul(weight).sum()
        loss1.backward()
        optim1.step()

        optim2.zero_grad()
        logits2 = net2(inten)
        loss2 = criteria2(logits2, lbs).mul(weight).sum()
        loss2.backward()
        optim2.step()

        #  o1 = one_hot
        #  o2 = grads['one_hot']
        #  print((o1 - o2).abs().max())
        #  print(o1.size())
        #  print(o2.size())

        with torch.no_grad():
            if (it+1) % 50 == 0:
                print('iter: {}, ================='.format(it+1))
                #  print(net1.fc.weight.numel())
                print('fc weight: ', torch.max(torch.abs(net1.fc.weight - net2.fc.weight)).item())

                print('conv1 weight: ', torch.max(torch.abs(net1.conv1.weight - net2.conv1.weight)).item())
                print('loss: ', loss1.item() - loss2.item())
