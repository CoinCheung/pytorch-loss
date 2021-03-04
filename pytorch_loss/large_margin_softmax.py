
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.amp as amp

'''
proposed in the BMVC2019 paper: [Large Margin in Softmax Cross-Entropy Loss
link to paper](https://staff.aist.go.jp/takumi.kobayashi/publication/2019/BMVC2019.pdf)
'''

##
# version 1: use torch.autograd
class LargeMarginSoftmaxV1(nn.Module):

    def __init__(self, lam=0.3, reduction='mean', ignore_index=255):
        super(LargeMarginSoftmaxV1, self).__init__()
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.lam = lam
        self.ce_crit = nn.CrossEntropyLoss(
                reduction='none', ignore_index=ignore_index)


    def forward(self, logits, label):
        '''
        Same usage method as nn.CrossEntropyLoss:
            >>> criteria = LargeMarginSoftmaxV1()
            >>> logits = torch.randn(8, 19, 384, 384) # nchw, float/half
            >>> lbs = torch.randint(0, 19, (8, 384, 384)) # nhw, int64_t
            >>> loss = criteria(logits, lbs)
        '''
        # overcome ignored label
        logits = logits.float()
        logits.retain_grad()
        logits.register_hook(lambda grad: grad)
        with torch.no_grad():
            num_classes = logits.size(1)
            coeff = 1. / (num_classes - 1.)
            lb = label.clone().detach()
            mask = label == self.ignore_index
            lb[mask] = 0
            idx = torch.zeros_like(logits).scatter_(1, lb.unsqueeze(1), 1.)

        lgts = logits - idx * 1.e6
        q = lgts.softmax(dim=1)
        q = q * (1. - idx)

        log_q = lgts.log_softmax(dim=1)
        log_q = log_q * (1. - idx)
        mg_loss = ((q - coeff) * log_q) * (self.lam / 2)
        mg_loss = mg_loss * (1. - idx)
        mg_loss = mg_loss.sum(dim=1)

        ce_loss = self.ce_crit(logits, label)
        loss = ce_loss + mg_loss
        loss = loss[mask == 0]

        if self.reduction == 'mean':
            loss = loss.mean()
        if self.reduction == 'sum':
            loss = loss.sum()

        return loss


##
# version 2: user derived grad computation
class LargeMarginSoftmaxV2(nn.Module):

    def __init__(self, lam=0.3, reduction='mean', ignore_index=255):
        super(LargeMarginSoftmaxV2, self).__init__()
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.lam = lam

    def forward(self, logits, labels):
        '''
        Same usage method as nn.CrossEntropyLoss:
            >>> criteria = LargeMarginSoftmaxV2()
            >>> logits = torch.randn(8, 19, 384, 384) # nchw, float/half
            >>> lbs = torch.randint(0, 19, (8, 384, 384)) # nhw, int64_t
            >>> loss = criteria(logits, lbs)
        '''
        logits = logits.float()
        mask = labels == self.ignore_index
        lb = labels.clone().detach()
        lb[mask] = 0
        loss = LargeMarginSoftmaxFuncV2.apply(logits, lb, self.lam)
        loss = loss[mask == 0]
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss



class LargeMarginSoftmaxFuncV2(torch.autograd.Function):

    @staticmethod
    @amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, logits, labels, lam=0.3):
        num_classes = logits.size(1)
        coeff = 1. / (num_classes - 1.)
        idx = torch.zeros_like(logits).scatter_(1, labels.unsqueeze(1), 1.)

        lgts = logits.clone()
        lgts[idx.bool()] = -1.e6
        q = lgts.softmax(dim=1)
        log_q = lgts.log_softmax(dim=1)
        losses = q.sub_(coeff).mul_(log_q).mul_(lam / 2.)
        losses[idx.bool()] = 0

        losses = losses.sum(dim=1).add_(F.cross_entropy(logits, labels, reduction='none'))

        ctx.variables = logits, labels, idx, coeff, lam
        return losses

    @staticmethod
    @amp.custom_bwd
    def backward(ctx, grad_output):
        '''
        compute gradient
        '''
        logits, labels, idx, coeff, lam = ctx.variables
        num_classes = logits.size(1)

        p = logits.softmax(dim=1)
        lgts = logits.clone()
        lgts[idx.bool()] = -1.e6
        q = lgts.softmax(dim=1)
        qx = q * lgts
        qx[idx.bool()] = 0

        grad = qx + q - q * qx.sum(dim=1).unsqueeze(1) - coeff
        grad = grad * lam / 2.
        grad[idx.bool()] = -1
        grad = grad + p

        grad.mul_(grad_output.unsqueeze(1))

        return grad, None, None



#
#  version 3: implement wit cpp/cuda to save memory and accelerate
class LargeMarginSoftmaxV3(nn.Module):

    def __init__(self, lam=0.3, reduction='mean', ignore_index=255):
        super(LargeMarginSoftmaxV3, self).__init__()
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.lam = lam

    def forward(self, logits, labels):
        '''
        Same usage method as nn.CrossEntropyLoss:
            >>> criteria = LargeMarginSoftmaxV3()
            >>> logits = torch.randn(8, 19, 384, 384) # nchw, float/half
            >>> lbs = torch.randint(0, 19, (8, 384, 384)) # nhw, int64_t
            >>> loss = criteria(logits, lbs)
        '''
        logits = logits.float()
        losses = LargeMarginSoftmaxFuncV3.apply(
                logits, labels, self.lam, self.ignore_index)

        if self.reduction == 'mean':
            n_valid = (labels != self.ignore_index).sum()
            losses = losses.sum() / n_valid
        elif self.reduction == 'sum':
            losses = losses.sum()
        return losses


import large_margin_cpp
class LargeMarginSoftmaxFuncV3(torch.autograd.Function):
    '''
    use cpp/cuda to accelerate and shrink memory usage
    '''
    @staticmethod
    @amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, logits, labels, lam=0.3, ignore_index=255):
        losses = large_margin_cpp.l_margin_forward(logits, labels, lam, ignore_index)

        ctx.variables = logits, labels, lam, ignore_index
        return losses

    @staticmethod
    @amp.custom_bwd
    def backward(ctx, grad_output):
        '''
        compute gradient
        '''
        logits, labels, lam, ignore_index = ctx.variables
        grads = large_margin_cpp.l_margin_backward(
                logits, labels, lam, ignore_index)
        grads.mul_(grad_output.unsqueeze(1))

        return grads, None, None, None



if __name__ == '__main__':
    import torchvision
    import torch
    import numpy as np
    import random
    torch.manual_seed(15)
    random.seed(15)
    np.random.seed(15)
    torch.backends.cudnn.deterministic = True

    class Model(nn.Module):
        def __init__(self):
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
            self.out = nn.Conv2d(512, 3, 3, 1, 1)
        def forward(self, x):
            feat1 = self.conv1(x)
            feat2 = self.bn1(feat1)
            feat3 = self.relu(feat2)
            #  feat4 = self.maxpool(feat3)
            feat5 = self.layer1(feat3)
            feat6 = self.layer2(feat5)
            feat7 = self.layer3(feat6)
            feat8 = self.layer4(feat7)
            feat9 = self.out(feat8)
            out = feat9

            feat8.retain_grad()
            feat8.register_hook(lambda grad: grad*100000)
            return out, feat8
    net1 = Model()
    net2 = Model()
    from copy import deepcopy
    net2.load_state_dict(deepcopy(net1.state_dict()))

    #  criteria1 = nn.CrossEntropyLoss(reduction='mean')
    #  criteria2 = nn.CrossEntropyLoss(reduction='mean')
    criteria1 = LargeMarginSoftmaxV1(reduction='mean')
    criteria2 = LargeMarginSoftmaxV3(reduction='mean')
    net1.cuda()
    net2.cuda()
    net1.train()
    net2.train()
    criteria1.cuda()
    criteria2.cuda()

    optim1 = torch.optim.SGD(net1.parameters(), lr=1e-2)
    optim2 = torch.optim.SGD(net2.parameters(), lr=1e-2)

    bs = 32
    for it in range(1000):
        inten = torch.randn(bs, 3, 256, 256).cuda()
        lbs = torch.randint(0, 3, (bs, 16, 16)).cuda()
        lbs[16:, :, :10] = 255
        #  s = lbs.cpu().detach().numpy()
        #  np.save('../lb.npy', s)
        logits, feat = net1(inten.clone())
        loss1 = criteria1(logits, lbs.clone())#.div(bs * 8 * 8)
        optim1.zero_grad()
        loss1.backward()
        optim1.step()
        #  s = logits.cpu().detach().numpy()
        #  np.save('../logitsv2.npy', s)

        logits, feat = net2(inten.clone())
        loss2 = criteria2(logits, lbs.clone())#.div(bs * 8 * 8)
        optim2.zero_grad()
        loss2.backward()
        optim2.step()
        #  s = logits.cpu().detach().numpy()
        #  np.save('../logitsv3.npy', s)
        #  print(logits[0, :, 0, 0])
        #  print(lbs[0, 0, 0])

        #  print('net2.weight: ', net2.out.weight[0, 0, :, 0])
        #  net2.load_state_dict(net1.state_dict())
        with torch.no_grad():
            if (it+1) % 50 == 0:
            #  if True:
                #  print(loss1.item())
                #  print(loss2.item())
                #  break
                print('iter: {}, ================='.format(it+1))
                print('out.weight: ', torch.mean(torch.abs(net1.out.weight - net2.out.weight)).item())
                print('conv1.weight: ', torch.mean(torch.abs(net1.conv1.weight - net2.conv1.weight)).item())
                #  print(net1.out.weight.mean().item())
                #  print(net2.out.weight.mean().item())
                print('\nloss: ', loss1.item() - loss2.item())
