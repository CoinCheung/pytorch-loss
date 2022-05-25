'''
    Partial FC proposed in paper: https://arxiv.org/abs/2010.05222

    Notes:
    1. You should not wrap this in nn.DistributedDataParallel, since this is model parallelizatoin rather than data parallelization.
    2. You should use this in distributed mode for training.
    3. Num of ids should be dividable by total gpu number, for example, if you have 2 machines each with 8 gpus is 16, you should set `n_ids=16 x n`, where n is some integer according to your dataset.

    An example:
    ```
    dist.init_process_group(backend='nccl')

    model = define_model()
    model.cuda()
    model = nn.DistributedDataParallel(model) # model use distributed mode
    crit = PartialFCAMSoftmax(emb_dim=256, n_ids=10000, m=0.3, s=15) # crit not use distributed mode
    crit.cuda()

    params = list(model.parameters()) + list(crit.parameters()) # this loss has trainable fc
    optim = SGD(params, lr=1e-3)

    for ims, ids in dataloader:
        ims, ids = ims.cuda(), ids.cuda()
        embs = model(ims)
        loss = crit(ims, ids)

        optim.zero_grad()
        loss.backward()
        optim.step()
    ```
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.cuda.amp as amp



class PartialFCAMSoftmax(nn.Module):

    def __init__(self, emb_dim, n_ids=10, m=0.3, s=15, ratio=1., reduction='mean'):
        super(PartialFCAMSoftmax, self).__init__()
        assert dist.is_initialized(), "must initialize distributed before create this"
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        assert n_ids % world_size == 0, "number of ids should be divisible among gpus. please drop some ids, which should make trivial differences"
        self.n_ids = int(n_ids / world_size)
        self.emb_dim = emb_dim

        assert ratio > 0. and ratio <= 1., "sample ratio should be in (0., 1.]"
        self.m, self.s, self.ratio = m, s, ratio
        self.W = torch.nn.Parameter(torch.randn(emb_dim, self.n_ids), requires_grad=True)

        nn.init.xavier_normal_(self.W, gain=1)

        self.reduction = reduction


    def forward(self, x, lb):
        assert x.size()[0] == lb.size()[0]
        assert x.size()[1] == self.emb_dim

        x, lb = GatherFunction.apply(x, lb)

        if self.ratio < 1.:
            W, ind1, ind2, n_pos = SampleFunction.apply(self.W, lb, self.ratio)
        else:
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            W = self.W
            ind1 = lb.div(self.n_ids, rounding_mode='trunc') == rank
            ind2 = lb[ind1] % self.n_ids
            n_pos = ind1.sum()

        x_norm = F.normalize(x, dim=1)
        w_norm = F.normalize(W, dim=0)

        loss = PartialFCFunction.apply(x_norm, w_norm, ind1, ind2, n_pos, self.s, self.m)

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss


class GatherFunction(torch.autograd.Function):

    @staticmethod
    @amp.custom_fwd
    def forward(ctx, embs, lbs):
        world_size = dist.get_world_size()
        rank = dist.get_rank()

        N, C = embs.size()
        e_dtype = embs.dtype
        l_dtype = lbs.dtype
        device = embs.device

        embs = embs.contiguous()
        all_embs = torch.zeros(
                size=[N * world_size, C], dtype=e_dtype, device=device)
        dist.all_gather(list(all_embs.chunk(world_size, dim=0)), embs)
        lbs = lbs.contiguous()
        all_lbs = torch.zeros(
                size=[N * world_size], dtype=l_dtype, device=device)
        dist.all_gather(list(all_lbs.chunk(world_size, dim=0)), lbs)

        return all_embs, all_lbs


    @staticmethod
    @amp.custom_bwd
    def backward(ctx, grad_all_embs, grad_all_lbs):
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        N = int(grad_all_embs.size(0) / world_size)
        grads_embs = grad_all_embs[rank * N: (rank + 1) * N]
        return grads_embs, None


class SampleFunction(torch.autograd.Function):

    @staticmethod
    @amp.custom_fwd
    def forward(ctx, W, lb, ratio):
        assert ratio < 1., 'do not call this unless ratio should less than 1.'
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        device = lb.device

        Wshape = W.size()
        n_ids = W.size(1)
        n_sample = int(n_ids * ratio) + 1

        # id pos and neg
        lb_unq = lb.unique(sorted=True)
        pos_ind1 = lb_unq.div(n_ids, rounding_mode='trunc') == rank
        pos_ind2 = lb_unq[pos_ind1] % n_ids
        id_n_pos = pos_ind1.sum()
        id_n_neg = max(0, n_sample - id_n_pos)

        # label pos and neg
        ind1 = lb.div(n_ids, rounding_mode='trunc') == rank
        ind2 = lb[ind1] % n_ids
        n_pos = ind1.sum()

        # no need to sample
        if id_n_pos == n_ids:
            keep_ind = torch.arange(n_ids, device=device)
            ctx.vars = keep_ind, Wshape
            return W, ind1, ind2, n_pos

        # sample ids
        if id_n_neg == 0:
            keep_ind = ind2
        elif id_n_pos == 0:
            keep_ind = torch.randperm(n_ids, device=device)[:id_n_neg]
        else:
            neg_mask = torch.ones(n_ids, device=device)
            neg_mask[pos_ind2] = 0
            neg_ind = neg_mask.nonzero()[:, 0]
            neg_mask = torch.randperm(neg_ind.size(0), device=device)[:id_n_neg]
            neg_ind = neg_ind[neg_mask]
            keep_ind = torch.cat([pos_ind2, neg_ind], dim=0)
        W = W[:, keep_ind]

        # map ind2 after sample
        if n_pos > 0:
            ind2 = (ind2.unsqueeze(1) == pos_ind2.unsqueeze(0)).nonzero()[:, 1]

        ctx.vars = keep_ind, Wshape
        return W, ind1, ind2, n_pos


    @staticmethod
    @amp.custom_bwd
    def backward(ctx, grad_W, grad_ind1, grad_ind2, grad_n_pos):
        keep_ind, Wshape = ctx.vars
        grad = torch.zeros(Wshape, dtype=grad_W.dtype, device=grad_W.device)
        grad[:, keep_ind] = grad_W
        return grad, None, None


class PartialFCFunction(torch.autograd.Function):

    @staticmethod
    @amp.custom_fwd
    def forward(ctx, all_embs, W, ind1, ind2, n_pos, s, m):

        assert all_embs.size(1) == W.size(0)
        N, C = all_embs.size()
        n_ids = W.size(1)
        e_dtype = all_embs.dtype
        device = all_embs.device
        world_size = dist.get_world_size()
        rank = dist.get_rank()

        logits = torch.einsum('ab,bc->ac', all_embs, W)

        # add amsoftmax margin and scale
        if n_pos > 0:
            logits[ind1, ind2] -= m
        logits *= s

        # we use float32 to compute softmax ce, since too much ids would make exp sum overflow
        logits = logits.float()
        l_max = logits.max(dim=1, keepdim=True)[0]
        dist.all_reduce(l_max, dist.ReduceOp.MAX)
        logits -= l_max
        l_exp = logits.exp_()
        l_exp_sum = l_exp.sum(dim=1, keepdim=True)
        dist.all_reduce(l_exp_sum, dist.ReduceOp.SUM)
        softmax = l_exp.div_(l_exp_sum)
        softmax = softmax.to(e_dtype)

        # nll loss
        loss = torch.zeros(all_embs.size(0), dtype=e_dtype, device=device)
        if n_pos > 0:
            prob = softmax[ind1, ind2]
            loss[ind1] = prob.log().neg()
        dist.all_reduce(loss, dist.ReduceOp.SUM)

        ctx.vars = softmax, ind1, ind2, n_pos, s, W, all_embs

        return loss

    @staticmethod
    @amp.custom_bwd
    def backward(ctx, grad_output):
        softmax, ind1, ind2, n_pos, s, W, all_embs = ctx.vars

        grads = softmax
        if n_pos > 0:
            grads[ind1, ind2] -= 1
        grads *= grad_output.view(-1, 1)

        grads *= s

        # we reduce sum grads of embs, but not W, according to chain rule
        grads_embs = torch.einsum('ac,bc->ab', grads, W)
        dist.all_reduce(grads_embs, dist.ReduceOp.SUM)

        grads_W = torch.einsum('ac,ab->cb', all_embs, grads)

        return grads_embs, grads_W, None, None, None, None, None

