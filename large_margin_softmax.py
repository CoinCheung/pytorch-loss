
import torch
import torch.nn as nn
import torch.nn.functional as F

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
        args: logits: tensor of shape (N, C, H, W, ...)
        args: label: tensor of shape(N, H, W, ...)
        '''
        # overcome ignored label
        logits = logits.float()
        logits.retain_grad()
        logits.register_hook(lambda grad: grad)
        #  n, c, h, w = logits.size()
        with torch.no_grad():
            num_classes = logits.size(1)
            coeff = 1. / (num_classes - 1.)
            lb = label.clone().detach()
            mask = label == self.ignore_index
            lb[mask] = 0
            idx = torch.zeros_like(logits).scatter_(1, lb.unsqueeze(1), 1.)
            #  idx = torch.ones((n * h * w, c), dtype=torch.int32).detach()
            #  idx = idx.to(logits.device).scatter_(1, lb.view(-1, 1), 0.)#.view(-1, c)
            #  print(idx.size())

        #  print(logits.size())
        #  print(logits.permute(0, 2, 3, 1).size())

        #  lgts = logits.clone()
        #  lgts[idx.bool()] = -1.e6
        lgts = logits - idx * 1.e6
        #  lgts.retain_grad()
        #  lgts.register_hook(lambda grad: grad)

        #  lgts = logits.permute(0, 2, 3, 1).view(-1, c)
        #  lgts = lgts[idx.bool()].view(n, h, w, -1).permute(0, 3, 1, 2)
        q = lgts.softmax(dim=1)
        q = q * (1. - idx)
        #  q.retain_grad()
        #  q.register_hook(lambda grad: grad)

        log_q = lgts.log_softmax(dim=1)
        log_q = log_q * (1. - idx)
        #  log_q.retain_grad()
        #  log_q.register_hook(lambda grad: grad)
        mg_loss = ((q - coeff) * log_q) * (self.lam / 2)
        #  mg_loss[idx.bool()] = 0
        mg_loss = mg_loss * (1. - idx)
        #  print(mg_loss[0, :, 0, 0])
        mg_loss = mg_loss.sum(dim=1)

        ce_loss = self.ce_crit(logits, label)
        loss = ce_loss + mg_loss
        loss = loss[mask == 0]

        #  print(label[0, 0, 0])
        #  print(idx[0, :, 0, 0])
        #  print(logits[0, :, 0, 0])
        #  print(lgts[0, :, 0, 0])
        #  max_lb = logits.max(dim=1)[0]
        #  max_no_lb =lgts.max(dim=1)[0]
        #  sum_lb = (logits - max_lb.unsqueeze(1)).exp().sum(dim=1)
        #  sum_no_lb = (lgts - max_no_lb.unsqueeze(1)).exp().sum(dim=1)
        #  print(max_no_lb[0, 0, 0])
        #  print(max_lb[0, 0, 0])
        #  print(sum_no_lb[0,0,0])
        #  print(sum_lb[0,0,0])
        #  print(q[0, :, 0, 0])
        #  print(mg_loss[0, 0, 0])

        #  print((logits - max_lb.unsqueeze(1))[0,0,0,0])
        #  print((lgts - max_no_lb.unsqueeze(1)).exp()[0, 0, 0, 0])
        #  print((lgts - max_no_lb.unsqueeze(1)).exp()[0, 1, 0, 0])
        #  print(max_lb.size())

        num = loss.numel()
        print('num: ', num)
        if self.reduction == 'mean':
            loss = loss.mean()
        if self.reduction == 'sum':
            loss = loss.sum()
        loss.backward()

        #  qx = q * lgts
        #  qx[idx.bool()] = 0
        #  print('loss1')
        #  print(qx[0, :, 0, 0])
        #  print(qx.sum(dim=1).unsqueeze(1)[0, :, 0, 0])
        #  print(logits.softmax(1)[0, :, 0, 0])
        #  print(logits[0, :, 0, 0])
        #  print(lgts.grad[0, :, 0, 0])
        print('logits.grad1', logits.grad[0, :, 0, 0]*1000)
        #  print(log_q.grad[0, :, 0, 0])
        #  print(q.grad[0, :, 0, 0])
        #  term = 0.5 * self.lam * (log_q + 1 - coeff / q) / num
        #  term = 0.5 * self.lam * (log_q) / num
        #  term = 0.5 * self.lam * (q - coeff) / num
        ## grad of logits
        #  term = logits.softmax(1) / num + lgts.grad
        #  term[idx.bool()] -= 1 / num
        #  print(term[0, :, 0, 0])
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
        loss.backward()
        return loss



class LargeMarginSoftmaxFuncV2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, labels, lam=0.3):
        num_classes = logits.size(1)
        coeff = 1. / (num_classes - 1.)
        lam = lam / 2
        idx = torch.zeros_like(logits).scatter_(1, labels.unsqueeze(1), 1.)

        lgts = logits.clone()
        lgts[idx.bool()] = -1.e6
        q = lgts.softmax(dim=1)
        log_q = lgts.log_softmax(dim=1)
        losses = q.sub_(coeff).mul_(log_q).mul_(lam)
        losses[idx.bool()] = 0

        losses = losses.sum(dim=1).add_(F.cross_entropy(logits, labels, reduction='none'))

        ctx.variables = logits, labels, idx, coeff, lam
        return losses


    #  @staticmethod
    #  def backward(ctx, grad_output):
    #      '''
    #      compute gradient
    #      '''
    #      logits, labels, idx, coeff, lam = ctx.variables
    #      num_classes = logits.size(1)
    #
    #      p = logits.softmax(dim=1)
    #      lgts = logits.clone()
    #      lgts[idx.bool()] = -1.e6
    #      q = lgts.softmax(dim=1)
    #      log_q = lgts.log_softmax(dim=1)
    #      q_log_q = q * log_q
    #      q_log_q[idx.bool()] = 0
    #
    #      #  grad = q_log_q + q - q_log_q.sum(dim=1).unsqueeze(1) - coeff
    #      #  grad = grad * lam
    #
    #      grad = log_q + 1. - q_log_q.sum(dim=1).unsqueeze(1)
    #      grad = grad * q - coeff
    #      grad = grad * lam
    #
    #      print('loss2')
    #      print(logits[0, :, 0, 0])
    #
    #      grad[idx.bool()] = -1
    #      #  print(grad[0, :, 0, 0])
    #      grad = grad + p
    #      print((grad * grad_output.unsqueeze(1))[0, :, 0, 0])
    #      p[idx.bool()] -= 1
    #      print(p[0, :, 0, 0] / 1024)
    #      print((p * grad_output.unsqueeze(1))[0, :, 0, 0])
    #      #  print(grad[0, :, 0, 0])
    #
    #
    #      #  grad.add_(p)
    #
    #      #  grad = (q * lgts).sum(dim=1).unsqueeze(1).neg_().add_(lgts).add_(1).mul_(q).sub_(coeff).mul_(lam).add_(p)
    #      #  grads[idx.bool()] -= 1
    #
    #      grad.mul_(grad_output.unsqueeze(1))
    #      print(grad[0, :, 0, 0])
    #      #  print(grad_output.unsqueeze(1)[0, :, 0, 0].item())
    #      #  print(grad[0, :, 0, 0])
    #      return grad, None, None


    #  @staticmethod
    #  def backward(ctx, grad_output):
    #      '''
    #      compute gradient
    #      '''
    #      logits, labels, idx, coeff, lam = ctx.variables
    #      num_classes = logits.size(1)
    #
    #      lgts = logits.clone()
    #      lgts[idx.bool()] = -1.e6
    #      q = lgts.softmax(dim=1)
    #      log_q = lgts.log_softmax(dim=1)
    #
    #      s = (q * log_q)
    #      s.add_(q)
    #      s.sub_(coeff)
    #      s[idx.bool()] = 0
    #      s = s.sum(dim=1).unsqueeze(1)
    #      #  print(s[0, :, 0, 0])
    #
    #      #  print(log_q[0, :, 0, 0])
    #      grad = log_q + 1 - s
    #      #  print(grad[0, :, 0, 0])
    #      grad.mul_(q)
    #      #  print(q[0, :, 0, 0])
    #      #  print(grad[0, :, 0, 0])
    #      grad.sub_(coeff)
    #      #  print(grad[0, :, 0, 0])
    #      grad.mul_(lam)
    #      #  print(grad[0, :, 0, 0])
    #
    #      #
    #      #  grad.mul_(q)
    #      #  grad.add_(q)
    #      #  grad.sub_(coeff)
    #      #  q.mul_(s)
    #      #  grad.sub_(q)
    #      #  grad.mul_(lam)
    #      grad[idx.bool()] = -1
    #      #  print(grad[0, :, 0, 0])
    #      p = logits.softmax(dim=1)
    #      grad.add_(p)
    #      #  print(grad[0, :, 0, 0])
    #
    #
    #      #  grad.add_(p)
    #
    #      #  grad = (q * lgts).sum(dim=1).unsqueeze(1).neg_().add_(lgts).add_(1).mul_(q).sub_(coeff).mul_(lam).add_(p)
    #      #  grads[idx.bool()] -= 1
    #
    #      grad.mul_(grad_output.unsqueeze(1))
    #      #  print(grad_output.unsqueeze(1)[0, :, 0, 0].item())
    #      #  print(grad[0, :, 0, 0])
    #      return grad, None, None

    @staticmethod
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
        #  qx = q * lgts.max(dim=1, keepdim=True)[0]
        qx[idx.bool()] = 0

        grad = qx + q - q * qx.sum(dim=1).unsqueeze(1) - coeff
        grad = grad * lam
        grad[idx.bool()] = -1
        grad = grad + p

        grad.mul_(grad_output.unsqueeze(1))

        return grad, None, None




##
# version 3: implement wit cpp/cuda to save memory and accelerate
class LargeMarginSoftmaxV3(nn.Module):

    def __init__(self, lam=0.3, reduction='mean', ignore_index=255):
        super(LargeMarginSoftmaxV3, self).__init__()
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.lam = lam

    def forward(self, logits, labels):
        '''
        args: logits: tensor of shape (N, C, H, W, ...)
        args: label: tensor of shape(N, H, W, ...)
        '''
        logits = logits.float()
        losses = LargeMarginSoftmaxFuncV3.apply(
                logits, labels, self.lam, self.ignore_index)

        #  logits.retain_grad()
        #  logits.register_hook(lambda grad: grad)

        if self.reduction == 'mean':
            n_valid = (labels != self.ignore_index).sum()
            losses = losses.sum() / n_valid
        elif self.reduction == 'sum':
            losses = losses.sum()
        losses.backward()
        #  print('logits.grad2', logits.grad[0, :, 0, 0]*1000)
        return losses


import large_margin_cpp
class LargeMarginSoftmaxFuncV3(torch.autograd.Function):
    '''
    use cpp/cuda to accelerate and shrink memory usage
    '''
    @staticmethod
    def forward(ctx, logits, labels, lam=0.3, ignore_index=255):
        losses = large_margin_cpp.l_margin_forward(logits, labels, lam, ignore_index)

        ctx.variables = logits, labels, lam, ignore_index
        return losses

    @staticmethod
    def backward(ctx, grad_output):
        '''
        compute gradient
        '''
        logits, labels, lam, ignore_index = ctx.variables
        grads = large_margin_cpp.l_margin_backward(
                grad_output, logits, labels, lam, ignore_index)
        print(grads[0, :, 0, 0])

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
    for it in range(2):
        inten = torch.randn(bs, 3, 256, 256).cuda()
        lbs = torch.randint(0, 3, (bs, 16, 16)).cuda()
        lbs[16:, :, :] = 255
        print('net1.weight: ', net1.out.weight[0, 0, :, 0])
        optim1.zero_grad()
        print('net1.weight: ', net1.out.weight[0, 0, :, 0])
        logits, feat = net1(inten.clone())
        print('net1.weight: ', net1.out.weight[0, 0, :, 0])
        print('logits1: ', logits[0, :, 0, 0])
        loss1 = criteria1(logits, lbs.clone())#.div(bs * 8 * 8)
        print('feat8.grad1', feat.grad[0, :4, 0, 0])
        #  loss1.backward()
        #  print(logits.grad[0, :, 0, 0])
        print('net1.weight: ', net1.out.weight[0, 0, :, 0])
        optim1.step()
        print('net1.weight: ', net1.out.weight[0, 0, :, 0])
        print('net2.weight: ', net2.out.weight[0, 0, :, 0])
        logits, feat = net2(inten.clone())
        print('logits2: ', logits[0, :, 0, 0])
        print('net2.weight: ', net2.out.weight[0, 0, :, 0])
        optim2.zero_grad()
        print('net2.weight: ', net2.out.weight[0, 0, :, 0])
        loss2 = criteria2(logits, lbs.clone())#.div(bs * 8 * 8)
        print('feat8.grad2', feat.grad[0, :4, 0, 0])
        print('net2.weight: ', net2.out.weight[0, 0, :, 0])
        #  loss2.backward()
        optim2.step()
        print('net2.weight: ', net2.out.weight[0, 0, :, 0])
        #  net2.load_state_dict(net1.state_dict())
        with torch.no_grad():
            if (it+1) % 1 == 0:
            #  if True:
                #  print(loss1.item())
                #  print(loss2.item())
                #  break
                print('iter: {}, ================='.format(it+1))
                print('out.weight: ', torch.mean(torch.abs(net1.out.weight - net2.out.weight)).item())
                print('conv1.weight: ', torch.mean(torch.abs(net1.conv1.weight - net2.conv1.weight)).item())
                #  print(net1.out.weight.mean().item())
                #  print(net2.out.weight.mean().item())
                #  print('\nloss: ', loss1.item() - loss2.item())
