
import torch
import torch.nn as nn
import torch.nn.functional as F

'''
proposed in the BMVC2019 paper: Large Margin in Softmax Cross-Entropy Loss
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
        args: logits: tensor of shape (N, C, H, W, ...)
        args: label: tensor of shape(N, H, W, ...)
        '''
        # overcome ignored label
        logits = logits.float()
        with torch.no_grad():
            bs, num_classes, *dims = logits.size()
            coeff = 1. / (num_classes - 1.)
            lb = label.clone().detach()
            mask = label == self.ignore_index
            lb[mask] = 0
            idx = torch.ones_like(logits).scatter_(1, lb.unsqueeze(1), 0)

        ce_loss = self.ce_crit(logits, label)
        lgts = logits[idx.bool()].view(bs, num_classes - 1, *dims)
        q = lgts.softmax(dim=1)
        log_q = lgts.log_softmax(dim=1)
        mg_loss = ((q - coeff) * log_q).sum(dim=1).mul_(self.lam / 2)

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
    '''
    proposed in the BMVC2019 paper: Large Margin in Softmax Cross-Entropy Loss
    '''
    def __init__(self, lam=0.3, reduction='mean', ignore_index=255):
        super(LargeMarginSoftmaxV2, self).__init__()
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.lam = lam

    def forward(self, logits, labels):
        '''
        args: logits: tensor of shape (N, C, H, W, ...)
        args: label: tensor of shape(N, H, W, ...)
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
    '''
    use cpp/cuda to accelerate and shrink memory usage
    '''
    @staticmethod
    def forward(ctx, logits, labels, lam=0.3):
        bs, num_classes, *dims = logits.size()
        coeff = 1. / (num_classes - 1.)
        idx = torch.ones_like(logits).scatter_(1, labels.unsqueeze(1), 0)
        lam = lam / 2

        lgts = logits[idx.bool()].view(bs, num_classes - 1, *dims)
        q = lgts.softmax(dim=1)
        log_q = lgts.log_softmax(dim=1)
        loss = log_q.mul_(q - coeff).sum(dim=1).mul_(lam)
        loss_ce = logits.log_softmax(dim=1).neg_()
        loss.add_(loss_ce[idx == 0].view(labels.size()))

        ctx.variables = logits, labels, idx, coeff, lam

        return loss

    @staticmethod
    def backward(ctx, grad_output):
        '''
        compute gradient
        '''
        logits, labels, idx, coeff, lam = ctx.variables
        bs, num_classes, *dims = logits.size()

        lgts = logits[idx.bool()].view(bs, num_classes - 1, *dims)
        q = lgts.softmax(dim=1)
        tmp = lgts.sub_((lgts * q).sum(dim=1, keepdim=True))
        tmp.mul_(q).add_(q.sub_(coeff)).mul_(lam)

        grads = logits.softmax(dim=1)
        grads[idx == 0] -= 1
        grads[idx.bool()] += tmp.view(-1)

        grads.mul_(grad_output.unsqueeze(1))
        return grads, None, None


class Try(nn.Module):
    '''
    proposed in the BMVC2019 paper: Large Margin in Softmax Cross-Entropy Loss
    '''
    def __init__(self, reduction='mean', ignore_index=255):
        super(Try, self).__init__()

    def forward(self, logits, labels):
        loss = logits.log_softmax(dim=1).neg()
        #  one_hot = torch.zeros_like(logits).scatter_(1, labels.unsqueeze(1), 1)
        idx = torch.ones_like(logits).scatter_(1, labels.unsqueeze(1), 0)
        loss = loss[idx == 0].view(labels.size())
        #  loss = (one_hot * loss).sum(dim=1)
        return loss.mean()



#  class LargeMarginSoftmaxV2(nn.Module):
#      '''
#      proposed in the BMVC2019 paper: Large Margin in Softmax Cross-Entropy Loss
#      '''
#      def __init__(self, lam=0.3, reduction='mean', ignore_index=255):
#          super(LargeMarginSoftmaxV2, self).__init__()
#          self.reduction = reduction
#          self.ignore_index = ignore_index
#          self.lam = lam
#          self.ce_crit = nn.CrossEntropyLoss(
#                  reduction='none', ignore_index=ignore_index)
#
#
#      def forward(self, logits, labels):
#          '''
#          args: logits: tensor of shape (N, C, H, W, ...)
#          args: label: tensor of shape(N, H, W, ...)
#          '''
#          logits = logits.float()
#          mask = labels != self.ignore_index
#          loss_ce = self.ce_crit(logits, labels)
#          loss_mg = LargeMarginSoftmaxFuncV2.apply(logits, labels, self.lam, self.ignore_index)
#          loss = loss_ce + loss_mg
#
#          loss = loss[mask]
#          if self.reduction == 'mean':
#              loss = loss.mean()
#          elif self.reduction == 'sum':
#              loss = loss.sum()
#          return loss
#
#
#  class LargeMarginSoftmaxFuncV2(torch.autograd.Function):
#      '''
#      use cpp/cuda to accelerate and shrink memory usage
#      '''
#      @staticmethod
#      def forward(ctx, logits, labels, lam=0.3, ignore_index=255):
#          bs, num_classes, *dims = logits.size()
#          coeff = 1. / (num_classes - 1.)
#          lb = labels.clone()
#          lb[lb == ignore_index] = 0
#          idx = torch.ones_like(logits).scatter_(1, lb.unsqueeze(1), 0)
#          lam = lam / 2
#
#          lgts = logits[idx.bool()].view(bs, num_classes - 1, *dims)
#          q = lgts.softmax(dim=1)
#          log_q = lgts.log_softmax(dim=1)
#          loss = ((q - coeff) * log_q).sum(dim=1).mul_(lam)
#
#          ctx.variables = logits, labels, idx, coeff, lam, ignore_index
#          return loss
#
#      @staticmethod
#      def backward(ctx, grad_output):
#          '''
#          compute gradient
#          '''
#          logits, labels, idx, coeff, lam, ignore_index = ctx.variables
#          bs, num_classes, *dims = logits.size()
#
#          lgts = logits[idx.bool()].view(bs, num_classes - 1, *dims)
#          q = lgts.softmax(dim=1)
#          tmp = lgts.sub_((lgts * q).sum(dim=1, keepdim=True))
#          tmp.mul_(q).add_(q.sub_(coeff)).mul_(lam)
#          grads = torch.zeros_like(logits)
#          grads[idx.bool()] = tmp.view(-1)
#          grads.mul_(grad_output.unsqueeze(1))
#          return grads, None, None, None



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
            feat = self.conv1(x)
            feat = self.bn1(feat)
            feat = self.relu(feat)
            feat = self.maxpool(feat)
            feat = self.layer1(feat)
            feat = self.layer2(feat)
            feat = self.layer3(feat)
            feat = self.layer4(feat)
            feat = self.out(feat)
            out = feat
            return out
    net1 = Model()
    net2 = Model()
    net2.load_state_dict(net1.state_dict())

    #  criteria1 = nn.CrossEntropyLoss(reduction='mean')
    #  criteria2 = Try()
    criteria1 = LargeMarginSoftmaxV1(reduction='mean')
    criteria2 = LargeMarginSoftmaxV2(reduction='mean')
    net1.cuda()
    net2.cuda()
    net1.train()
    net2.train()
    criteria1.cuda()
    criteria2.cuda()

    optim1 = torch.optim.SGD(net1.parameters(), lr=1e-2)
    optim2 = torch.optim.SGD(net2.parameters(), lr=1e-2)

    bs = 32
    for it in range(300000):
        inten = torch.randn(bs, 3, 256, 256).cuda()
        lbs = torch.randint(0, 3, (bs, 8, 8)).cuda()
        logits = net1(inten)
        loss1 = criteria1(logits, lbs)#.div(bs * 8 * 8)
        optim1.zero_grad()
        loss1.backward()
        optim1.step()
        logits = net2(inten)
        loss2 = criteria2(logits, lbs)#.div(bs * 8 * 8)
        optim2.zero_grad()
        loss2.backward()
        optim2.step()
        with torch.no_grad():
            if (it+1) % 50 == 0:
            #  if True:
                #  print(loss1.item())
                #  print(loss2.item())
                #  break
                print('iter: {}, ================='.format(it+1))
                print('out.weight: ', torch.mean(torch.abs(net1.out.weight - net2.out.weight)).item())
                print('conv1.weight: ', torch.mean(torch.abs(net1.conv1.weight - net2.conv1.weight)).item())
                print('loss: ', loss1.item() - loss2.item())
