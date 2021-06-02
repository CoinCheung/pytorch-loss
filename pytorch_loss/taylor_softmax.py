

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.amp as amp


'''
proposed in this paper: [Exploring Alternatives to Softmax Function](https://arxiv.org/pdf/2011.11538.pdf)
'''


##
# functions
import taylor_softmax_cpp
class TaylorSoftmaxFunc(torch.autograd.Function):
    '''
    use cpp/cuda to accelerate and shrink memory usage
    '''
    @staticmethod
    @amp.custom_fwd
    def forward(ctx, feat, dim=1, n=2, use_log=False):
        ctx.vars = feat, dim, n, use_log
        return taylor_softmax_cpp.taylor_softmax_forward(feat, dim, n, use_log)

    @staticmethod
    @amp.custom_bwd
    def backward(ctx, grad_output):
        '''
        compute gradient of focal loss
        '''
        feat, dim, n, use_log = ctx.vars
        return taylor_softmax_cpp.taylor_softmax_backward(grad_output, feat, dim, n, use_log), None, None, None


def taylor_softmax_v1(x, dim=1, n=4, use_log=False):
    assert n % 2 == 0 and n > 0
    fn = torch.ones_like(x)
    denor = 1.
    for i in range(1, n + 1):
        denor *= i
        fn = fn + x.pow(i) / denor
    out = fn / fn.sum(dim=dim, keepdims=True)
    if use_log: out = out.log()
    return out


def taylor_softmax_v3(inten, dim=1, n=4, use_log=False):
    assert n % 2 == 0 and n > 0
    return TaylorSoftmaxFunc.apply(inten, dim, n, use_log)



### TaylorSoftmax
##
# version 1: use torch.autograd
class TaylorSoftmaxV1(nn.Module):
    '''
    This is the autograd version
    '''
    def __init__(self, dim=1, n=2):
        super(TaylorSoftmaxV1, self).__init__()
        self.dim = dim
        self.n = n

    def forward(self, x):
        '''
        usage similar to nn.Softmax:
            >>> mod = TaylorSoftmaxV1(dim=1, n=4)
            >>> inten = torch.randn(1, 32, 64, 64)
            >>> out = mod(inten)
        '''
        return taylor_softmax_v1(x, self.dim, self.n, use_log=False)


##
# version 3: use cuda
class TaylorSoftmaxV3(nn.Module):
    '''
    This is the autograd version
    '''
    def __init__(self, dim=1, n=2):
        super(TaylorSoftmaxV3, self).__init__()
        assert n % 2 == 0 and n > 0
        self.dim = dim
        self.n = n

    def forward(self, x):
        '''
        usage similar to nn.Softmax:
            >>> mod = TaylorSoftmaxV3(dim=1, n=4)
            >>> inten = torch.randn(1, 32, 64, 64)
            >>> out = mod(inten)
        '''
        return taylor_softmax_v3(x, self.dim, self.n, use_log=False)



### LogSoftmax
##
# version 1: use torch.autograd
class LogTaylorSoftmaxV1(nn.Module):
    '''
    This is the autograd version
    '''
    def __init__(self, dim=1, n=2):
        super(LogTaylorSoftmaxV1, self).__init__()
        assert n % 2 == 0
        self.dim = dim
        self.n = n

    def forward(self, x):
        '''
        usage similar to nn.Softmax:
            >>> mod = LogTaylorSoftmaxV1(dim=1, n=4)
            >>> inten = torch.randn(1, 32, 64, 64)
            >>> out = mod(inten)
        '''
        return taylor_softmax_v1(x, self.dim, self.n, use_log=True)


##
# version 3: use cuda
class LogTaylorSoftmaxV3(nn.Module):
    '''
    This is the autograd version
    '''
    def __init__(self, dim=1, n=2):
        super(LogTaylorSoftmaxV3, self).__init__()
        assert n % 2 == 0
        self.dim = dim
        self.n = n

    def forward(self, x):
        '''
        usage similar to nn.Softmax:
            >>> mod = LogTaylorSoftmaxV3(dim=1, n=4)
            >>> inten = torch.randn(1, 32, 64, 64)
            >>> out = mod(inten)
        '''
        return taylor_softmax_v3(x, self.dim, self.n, use_log=True)



### SoftmaxCrossEntropy
##
# version 1: use torch.autograd
class TaylorCrossEntropyLossV1(nn.Module):
    '''
    This is the autograd version
    '''
    def __init__(self, n=2, ignore_index=-1, reduction='mean'):
        super(TaylorCrossEntropyLossV1, self).__init__()
        assert n % 2 == 0
        self.taylor_softmax = LogTaylorSoftmaxV1(dim=1, n=n)
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, logits, labels):
        '''
        usage similar to nn.CrossEntropyLoss:
            >>> crit = TaylorCrossEntropyLossV1(n=4)
            >>> inten = torch.randn(1, 10, 64, 64)
            >>> label = torch.randint(0, 10, (1, 64, 64))
            >>> out = crit(inten, label)
        '''
        log_probs = self.taylor_softmax(logits)
        loss = F.nll_loss(log_probs, labels, reduction=self.reduction,
                ignore_index=self.ignore_index)
        return loss

##
# version 3: use cuda
class TaylorCrossEntropyLossV3(nn.Module):
    '''
    This is the autograd version
    '''
    def __init__(self, n=2, ignore_index=-1, reduction='mean'):
        super(TaylorCrossEntropyLossV3, self).__init__()
        assert n % 2 == 0
        self.taylor_softmax = LogTaylorSoftmaxV3(dim=1, n=n)
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, logits, labels):
        '''
        usage similar to nn.CrossEntropyLoss:
            >>> crit = TaylorCrossEntropyLossV3(n=4)
            >>> inten = torch.randn(1, 10, 64, 64)
            >>> label = torch.randint(0, 10, (1, 64, 64))
            >>> out = crit(inten, label)
        '''
        log_probs = self.taylor_softmax(logits)
        loss = F.nll_loss(log_probs, labels, reduction=self.reduction,
                ignore_index=self.ignore_index)
        return loss


if __name__ == '__main__':
    import numpy as np
    import torchvision
    torch.backends.cudnn.deterministic = True
    #  tsoftmax = TaylorSoftmaxV3(dim=0, n=4)
    #  inten = torch.randn(3, 4, 5, 6).cuda()
    #  out = tsoftmax(inten)
    #  print(out.size())
    #  print(out)
    #  print(out[:, 0, 0, :4])
    #  print(out[:, 0, 0, :4].sum(dim=0))


    class Model(nn.Module):
        def __init__(self, softmax='v1'):
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
            self.fc = nn.Conv2d(512, 19 * 32 * 32, 3, 1, 1)
            self.upsample = nn.PixelShuffle(32)
            self.softmax = softmax
            if softmax == 'v1':
                obj = LogTaylorSoftmaxV1
                self.softmax1 = obj(dim=0, n=2)
                self.softmax2 = obj(dim=1, n=4)
                self.softmax3 = obj(dim=2, n=6)
                self.softmax4 = obj(dim=3, n=8)
            else:
                obj = LogTaylorSoftmaxV3
                self.softmax1 = obj(dim=0, n=2)
                self.softmax2 = obj(dim=1, n=4)
                self.softmax3 = obj(dim=2, n=6)
                self.softmax4 = obj(dim=3, n=8)

        def forward(self, x):
            feat = self.conv1(x)
            feat = self.bn1(feat)
            feat = self.relu(feat)
            feat = self.maxpool(feat)
            feat = self.layer1(feat)
            feat = self.softmax1(feat)
            feat = self.layer2(feat)
            #  arr = feat.cpu().detach().numpy().tofile('tmp.npy')
            #  size = feat.size()
            #  arr = np.fromfile('tmp.npy', dtype=np.float32)
            #  feat = torch.from_numpy(arr).cuda().view(size)
            feat = self.softmax2(feat)
            feat = self.layer3(feat)
            feat = self.softmax3(feat)
            feat = self.layer4(feat)
            feat = self.softmax4(feat)
            feat = self.fc(feat)
            out = self.upsample(feat)
            #  out = F.interpolate(feat, x.size()[2:], mode='bilinear', align_corners=True)
            return out

    red = 'mean'
    bs = 64
    net1 = Model(softmax='v1')
    net2 = Model(softmax='v3')
    net2.load_state_dict(net1.state_dict())
    net1.cuda()
    net2.cuda()
    net1.train()
    net2.train()

    criteria1 = TaylorCrossEntropyLoss(n=4, ignore_index=255, reduction=red)
    criteria2 = TaylorCrossEntropyLoss(n=4, ignore_index=255, reduction=red)
    #  criteria1 = nn.CrossEntropyLoss(ignore_index=255)
    #  criteria2 = nn.CrossEntropyLoss(ignore_index=255)

    optim1 = torch.optim.SGD(net1.parameters(), lr=1e-2)
    optim2 = torch.optim.SGD(net2.parameters(), lr=1e-2)

    for it in range(5000):
        inten = torch.randn(bs, 3, 224, 224).cuda()
        lbs = torch.randint(0, 19, (bs, 224, 224)).cuda()
        lbs[1, 1, 1] = 255
        lbs[30, 3, 2:200] = 255
        lbs[18, 4:7, 8:200] = 255

        #  net2.load_state_dict(net1.state_dict())
        logits1 = net1(inten)
        logits2 = net2(inten)
        #  print('logits2.size(): ', logits2.size())
        #  print('torch.isnan(logits2.sum()): ', torch.isnan(logits2).sum())
        loss1 = criteria1(logits1, lbs)
        loss2 = criteria2(logits2, lbs)

        optim1.zero_grad()
        optim2.zero_grad()
        loss1.backward()
        loss2.backward()
        optim1.step()
        optim2.step()

        #  _ = input()
        with torch.no_grad():
            if (it+1) % 50 == 0:
            #  if True:
                print('iter: {}, ================='.format(it+1))
                print('out.weight: ', torch.max(torch.abs(net1.fc.weight - net2.fc.weight)).item())
                print('conv1.weight: ', torch.max(torch.abs(net1.conv1.weight - net2.conv1.weight)).item())
                print('\nloss: ', loss1.item() - loss2.item())

