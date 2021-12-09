'''
    This can be used to train self-supervised learning models such as simclr. The merits of this implementation is that we can fully make use of negative samples even in distributed training mode. For example, if you use 8 gpus and each gpus has batch size of 32 x 2(two views of one image), you will use total negative samples of 32 x 2 x 8 - 2 = 510 negative samples to train your simclr model.

    This implementation uses a "model distributed" method rather than "data distributed" method, so you should use this in distributed training mode, but not wrap this in pytorch nn.DistributedParallel module.

    An example of usage is like this:
    ```python
        # init distributed mode
        dist.init_process_group(backend='nccl')

        model = define_model()
        model = nn.DistributedDataParallel(model) # model use distributed mode
        crit = InfoNceDist(temper=0.1, margin=0.) # crit not use distributed mode
        model.cuda()
        crit.cuda()

        params = list(model.parameters()) + list(crit.parameters())
        optim = SGD(params, lr=1e-3)

        for (ims_view1, ims_view2), ids in dataloader:
            ims_view1, ims_view2 = ims_view1.cuda(), ims_view2.cuda()
            ids = ids.cuda()

            embs1 = model(ims_view1)
            embs2 = model(ims_view2)
            loss = crit(embs1, embs2)

            optim.zero_grad()
            loss.backward()
            optim.step()
        ```
    ```

    For details of simclr, please refer to the their paper: https://arxiv.org/pdf/2002.05709.pdf

    Please note that this is different from the info-nce used in moco series, where a negative queue is given. This works in simclr's way.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.cuda.amp as amp



class InfoNceDist(nn.Module):

    def __init__(self, temper=0.1, margin=0.):
        super(InfoNceDist, self).__init__()
        self.crit = nn.CrossEntropyLoss()
        # we use margin, but not use s, because temperature works in same way as s
        self.margin = margin
        self.temp_factor = 1. / temper

    def forward(self, embs1, embs2):
        '''
        embs1, embs2: n x c, one by one pairs
            1 positive, 2n - 2 negative
            distributed mode, no need to wrap with nn.DistributedParallel
        '''
        embs1 = F.normalize(embs1, dim=1)
        embs2 = F.normalize(embs2, dim=1)
        logits, labels = InfoNceFunction.apply(embs1, embs2, self.temp_factor, self.margin)
        loss = self.crit(logits, labels.detach())
        return loss


class InfoNceFunction(torch.autograd.Function):

    @staticmethod
    @amp.custom_fwd
    def forward(ctx, embs1, embs2, temper_factor, margin):
        assert embs1.size() == embs2.size()
        N, C = embs1.size()
        dtype = embs1.dtype
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        device = embs1.device

        # gather for negative
        all_embs1 = torch.zeros(
                size=[N * world_size, C], dtype=dtype).cuda(device)
        dist.all_gather(list(all_embs1.chunk(world_size, dim=0)), embs1)
        all_embs2 = torch.zeros(
                size=[N * world_size, C], dtype=dtype).cuda(device)
        dist.all_gather(list(all_embs2.chunk(world_size, dim=0)), embs2)
        all_embs = torch.cat([all_embs1, all_embs2], dim=0)
        embs12 = torch.cat([embs1, embs2], dim=0)

        logits = torch.einsum('ac,bc->ab', embs12, all_embs)
        # mask off one sample to itself
        inds1 = torch.arange(N * 2).cuda(device)
        inds2 = torch.cat([
            torch.arange(N) + rank * N,
            torch.arange(N) + (rank + world_size) * N
            ], dim=0).cuda(device)
        logits[inds1, inds2] = -10000. # such that exp should be 0

        # label: 0~N should be N * [rank, rank + 1], N~(2N-1) should be N * [world_size * rank, world_size * (rank + 1)]
        labels = inds2.view(2, -1).flip(dims=(0,)).reshape(-1)

        # subtract margin, apply temperature
        logits[inds1, labels] -= margin
        logits *= temper_factor

        ctx.vars = inds1, inds2, embs12, all_embs, temper_factor
        return logits, labels

    @staticmethod
    @amp.custom_bwd
    def backward(ctx, grad_logits, grad_label):
        inds1, inds2, embs12, all_embs, temper_factor = ctx.vars

        grad_logits = grad_logits * temper_factor

        grad_logits[inds1, inds2] = 0
        grad_embs12 = torch.einsum('ab,bc->ac', grad_logits, all_embs)
        grad_all_embs = torch.einsum('ab,ac->bc', grad_logits, embs12)

        rank = dist.get_rank()
        world_size = dist.get_world_size()
        N = int(all_embs.size(0) / (world_size * 2))
        grad_embs1 = grad_embs12[:N] + grad_all_embs[rank * N : (rank + 1) * N]
        grad_embs2 = grad_embs12[N:] + grad_all_embs[(rank + world_size) * N : (rank + world_size + 1) * N]

        return grad_embs1, grad_embs2, None, None
