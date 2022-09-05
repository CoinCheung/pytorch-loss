#! /usr/bin/python
# -*- encoding: utf-8 -*-

'''
My implementation of giou, diou, ciou function and their associated losses: GIOULoss, DIOULoss, CIOULoss.

The motivation of implementing this is that the paper of CIOU said they replace the term of `1/(h^2 + w^2)` with constant number of `1` during backward computation, but I searched github for a few minutes without finding this part of code. Maybe some people is interested in this, so I write one on my own.

Please be aware that I did not replace yolov5 ciou loss with this to test the performance difference, so I do not know whether this would bring improvements. I simply implement this following the paper formula.
'''

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.amp as amp


## GIOU loss is proposed here: https://arxiv.org/abs/1902.09630
class GIOULoss(nn.Module):

    def __init__(self, eps=1e-5, reduction='mean'):
        super(GIOULoss, self).__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, pr_bboxes, gt_bboxes):
        """
        pr_bboxes: tensor (-1, 4) xyxy, predicted bbox
        gt_bboxes: tensor (-1, 4) xyxy, ground truth bbox
        loss proposed in the paper of giou
        """
        giou = giou_func(gt_bboxes, pr_bboxes, self.eps)
        loss = 1. - giou
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'none':
            pass
        return loss


## DIOU loss is proposed here: https://arxiv.org/abs/1911.08287
class DIOULoss(nn.Module):

    def __init__(self, eps=1e-5, reduction='mean'):
        super(DIOULoss, self).__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, pr_bboxes, gt_bboxes):
        """
        pr_bboxes: tensor (-1, 4) xyxy, predicted bbox
        gt_bboxes: tensor (-1, 4) xyxy, ground truth bbox
        loss proposed in the paper of giou
        """
        diou = diou_func(gt_bboxes, pr_bboxes, self.eps)
        loss = 1. - diou
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'none':
            pass
        return loss


## CIOU loss is also proposed here: https://arxiv.org/abs/1911.08287
class CIOULoss(nn.Module):

    def __init__(self, eps=1e-5, reduction='sum'):
        super(CIOULoss, self).__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, pr_bboxes, gt_bboxes):
        """
        pr_bboxes: tensor (-1, 4) xyxy, predicted bbox
        gt_bboxes: tensor (-1, 4) xyxy, ground truth bbox
        loss proposed in the paper of giou
        """
        ciou = ciou_func(gt_bboxes, pr_bboxes, self.eps)
        loss = 1. - ciou
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'none':
            pass
        return loss


def iou_func(gt_bboxes, pr_bboxes, eps=1e-5):
    """
    input:
        gt_bboxes: tensor (N, 4) xyxy
        pr_bboxes: tensor (N, 4) xyxy
    output:
        gious: tensor (N, )
    """
    gt_area = (gt_bboxes[:, 2]-gt_bboxes[:, 0])*(gt_bboxes[:, 3]-gt_bboxes[:, 1])
    pr_area = (pr_bboxes[:, 2]-pr_bboxes[:, 0])*(pr_bboxes[:, 3]-pr_bboxes[:, 1])

    # iou
    lt = torch.max(gt_bboxes[:, :2], pr_bboxes[:, :2])
    rb = torch.min(gt_bboxes[:, 2:], pr_bboxes[:, 2:])
    wh = (rb - lt + eps).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]
    union = gt_area + pr_area - inter
    iou = inter / union
    return iou


def giou_func(gt_bboxes, pr_bboxes, eps=1e-5):
    """
    input:
        gt_bboxes: tensor (N, 4) xyxy
        pr_bboxes: tensor (N, 4) xyxy
    output:
        gious: tensor (N, )
    """
    iou = iou_func(gt_bboxes, pr_bboxes, eps)

    # enclosure
    lt = torch.min(gt_bboxes[:, :2], pr_bboxes[:, :2])
    rb = torch.max(gt_bboxes[:, 2:], pr_bboxes[:, 2:])
    wh = (rb - lt + eps).clamp(min=0)
    enclosure = wh[:, 0] * wh[:, 1]

    giou = iou - (enclosure - union) / enclosure
    return giou



def diou_func(gt_bboxes, pr_bboxes, eps=1e-5):
    """
    input:
        gt_bboxes: tensor (N, 4) xyxy
        pr_bboxes: tensor (N, 4) xyxy
    output:
        dious: tensor (N, )
    """
    iou = iou_func(gt_bboxes, pr_bboxes, eps)

    # center distance
    #  gt_cent_x = gt_bboxes[:, 0::2].mean(dim=-1, keepdims=True)
    #  gt_cent_y = gt_bboxes[:, 1::2].mean(dim=-1, keepdims=True)
    #  pr_cent_x = pr_bboxes[:, 0::2].mean(dim=-1, keepdims=True)
    #  pr_cent_y = pr_bboxes[:, 1::2].mean(dim=-1, keepdims=True)
    #  gt_cent = torch.cat([gt_cent_x, gt_cent_y], dim=-1)
    #  pr_cent = torch.cat([pr_cent_x, pr_cent_y], dim=-1)
    #  cent_dis = F.pairwise_distance(gt_cent, pr_cent)
    gt_cent_x = gt_bboxes[:, 0::2].mean(dim=-1)
    gt_cent_y = gt_bboxes[:, 1::2].mean(dim=-1)
    pr_cent_x = pr_bboxes[:, 0::2].mean(dim=-1)
    pr_cent_y = pr_bboxes[:, 1::2].mean(dim=-1)
    cent_dis = (gt_cent_x - pr_cent_x).pow(2.) + (gt_cent_y - pr_cent_y).pow(2.)

    # diag distance
    lt = torch.min(gt_bboxes[:, :2], pr_bboxes[:, :2])
    rb = torch.max(gt_bboxes[:, 2:], pr_bboxes[:, 2:])
    #  diag_dis = F.pairwise_distance(lt, rb)
    diag_dis = (lt - rb).pow(2).sum(dim=-1)

    # diou
    #  reg = (cent_dis / (diag_dis + eps)).pow(2.)
    reg = cent_dis / (diag_dis + eps)
    diou = iou - reg
    return diou


def ciou_func(gt_bboxes, pr_bboxes, eps=1e-5):
    """
    input:
        gt_bboxes: tensor (N, 4) xyxy
        pr_bboxes: tensor (N, 4) xyxy
    output:
        cious: tensor (N, )
    """
    diou = diou_func(gt_bboxes, pr_bboxes, eps)
    # ciou reg
    creg = CIOURegFunc.apply(gt_bboxes, pr_bboxes, eps)

    ciou = diou - creg
    return ciou


class CIOURegFunc(torch.autograd.Function):
    '''
    forward and backward of CIOU regularization term
    '''
    @staticmethod
    @amp.custom_fwd
    def forward(ctx, gt_bboxes, pr_bboxes, eps=1e-5):
        gt_w = gt_bboxes[:, 2] - gt_bboxes[:, 0]
        gt_h = gt_bboxes[:, 3] - gt_bboxes[:, 1]
        pr_w = pr_bboxes[:, 2] - pr_bboxes[:, 0]
        pr_h = pr_bboxes[:, 3] - pr_bboxes[:, 1]
        coef = 4. / (math.pi ** 2)
        atan_diff = torch.atan(gt_w / gt_h) - torch.atan(pr_w / pr_h)
        v = atan_diff.pow(2.)
        v = coef * v
        iou = iou_func(gt_bboxes, pr_bboxes, eps)
        alpha = v / (1 - iou + v)
        reg = alpha * v

        ## we compute gradient directly, since bbox does not use too much memory
        # grad of pred bbox
        #  h2_w2 = 1. / (pr_h.pow(2.) + pr_w.pow(2.)) # org grad
        h2_w2 = 1. # replace with 1 as proposed in paper
        dv = 2 * coef * atan_diff * h2_w2 * alpha
        # this is negative of paper formula, but I think this is the right way
        dv_dh = dv * pr_w
        dv_dw = -dv * pr_h
        dx1, dx2 = -dv_dw.view(-1, 1), dv_dw.view(-1, 1)
        dy1, dy2 = -dv_dh.view(-1, 1), dv_dh.view(-1, 1)
        d_pr_bbox = torch.cat([dx1, dy1, dx2, dy2], dim=-1)

        # grad of gt bbox
        #  h2_w2 = 1. / (gt_h.pow(2.) + gt_w.pow(2.)) # org grad
        h2_w2 = 1. # replace with 1 as proposed in paper
        dv = 2 * coef * atan_diff * h2_w2 * alpha
        dv_dh = dv * gt_w
        dv_dw = -dv * gt_h
        dx1, dx2 = -dv_dw.view(-1, 1), dv_dw.view(-1, 1)
        dy1, dy2 = -dv_dh.view(-1, 1), dv_dh.view(-1, 1)
        d_gt_bbox = -torch.cat([dx1, dy1, dx2, dy2], dim=-1)

        ctx.variables = d_gt_bbox, d_pr_bbox
        return reg

    @staticmethod
    @amp.custom_bwd
    def backward(ctx, grad_output):
        d_gt_bbox, d_pr_bbox = ctx.variables

        return d_gt_bbox, d_pr_bbox, None


if __name__ == '__main__':
    #  gt_bbox = torch.tensor([[1, 2, 3, 4]], dtype=torch.float32)
    #  pr_bbox = torch.tensor([[2, 3, 4, 5]], dtype=torch.float32)
    #  loss = generalized_iou_loss(gt_bbox, pr_bbox, reduction='none')
    #  print(loss)

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
            self.out = nn.Linear(512, 4*10)
        def forward(self, x):
            feat = self.conv1(x)
            feat = self.bn1(feat)
            feat = self.relu(feat)
            feat = self.maxpool(feat)
            feat = self.layer1(feat)
            feat = self.layer2(feat)
            feat = self.layer3(feat)
            feat = self.layer4(feat)
            feat = torch.mean(feat, dim=(2, 3))
            feat = self.out(feat)
            feat = feat.reshape(-1, 4)
            return feat
    net1 = Model()
    net2 = Model()
    net2.load_state_dict(net1.state_dict())

    net1.cuda()
    net2.cuda()
    net1.train()
    net2.train()
    net1.double()
    net2.double()

    optim1 = torch.optim.SGD(net1.parameters(), lr=1e-2)
    optim2 = torch.optim.SGD(net2.parameters(), lr=1e-2)
    criteria1 = CIOULoss()


    def ciou_func_v2(gt_bboxes, pr_bboxes, eps=1e-5):
        """
        input:
            gt_bboxes: tensor (N, 4) xyxy
            pr_bboxes: tensor (N, 4) xyxy
        output:
            cious: tensor (N, )
        """
        diou = diou_func(gt_bboxes, pr_bboxes, eps)
        # ciou reg
        gt_w = gt_bboxes[:, 2] - gt_bboxes[:, 0]
        gt_h = gt_bboxes[:, 3] - gt_bboxes[:, 1]
        pr_w = pr_bboxes[:, 2] - pr_bboxes[:, 0]
        pr_h = pr_bboxes[:, 3] - pr_bboxes[:, 1]
        coef = 4. / (math.pi ** 2)
        atan_diff = torch.atan(gt_w / gt_h) - torch.atan(pr_w / pr_h)
        v = atan_diff.pow(2.)
        v = coef * v
        iou = iou_func(gt_bboxes, pr_bboxes, eps)
        with torch.no_grad():
            alpha = v / (1 - iou + v)
        creg = alpha.detach() * v

        ciou = diou - creg
        return ciou

    for it in range(100):
        inten = torch.randn(4, 3, 112, 112).double().cuda()
        gt_bboxes = torch.randn(40, 4).double().cuda()
        gt_bboxes1 = gt_bboxes
        gt_bboxes2 = gt_bboxes

        out1 = net1(inten)
        out2 = net2(inten)
        #  bs = out1.size(0)
        #  gt_bboxes1 = out1[bs//2:]
        #  out1 = out1[:bs//2]
        #  gt_bboxes2 = out2[bs//2:]
        #  out2 = out2[:bs//2]

        #  loss1 = 1. - ciou_func(gt_bboxes1, out1)
        loss1 = criteria1(out1, gt_bboxes1)
        loss2 = 1. - ciou_func_v2(out2, gt_bboxes2)
        #  loss1 = 1. - diou_func(gt_bboxes1, out1)
        #  loss2 = 1. - diou_func(gt_bboxes2, out2)
        #  loss1 = loss1.sum()
        loss2 = loss2.sum()

        optim1.zero_grad()
        loss1.backward()
        optim1.step()

        optim2.zero_grad()
        loss2.backward()
        optim2.step()

        with torch.no_grad():
            if (it+1) % 5 == 0:
                print('iter: {}, ================='.format(it+1))
                print('out.weight: ', torch.mean(torch.abs(net1.out.weight - net2.out.weight)).item())
                print('conv1.weight: ', torch.mean(torch.abs(net1.conv1.weight - net2.conv1.weight)).item())
                print('loss: ', loss1.item() - loss2.item())
