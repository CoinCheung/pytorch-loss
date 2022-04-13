#! /usr/bin/python
# -*- encoding: utf-8 -*-

'''
    Proposed in this paper: https://arxiv.org/abs/2204.01509
'''


import torch
import torch.nn as nn
import torch.nn.functional as F


class GroupLoss(nn.Module):

    def __init__(self, in_feats=2048, n_ids=100, n_iters=2,
            n_lbs_per_cls=2, has_fc=True):
        super(GroupLoss, self).__init__()
        self.n_lbs_per_cls = n_lbs_per_cls
        self.n_iters = n_iters
        self.has_fc = has_fc

        self.clip = nn.ReLU(inplace=True)
        self.fc = nn.Identity()
        if has_fc: self.fc = nn.Linear(in_feats, n_ids)

    def forward(self, emb, lbs, logits=None):
        if self.has_fc: logits = self.fc(emb)

        n, c = emb.size()
        n_cls = logits.size()[1]
        device = logits.device

        # pearson matrix
        emb_norm = emb - emb.mean(dim=1, keepdims=True)
        emb_norm = F.normalize(emb_norm, dim=1)
        W = torch.einsum('ab,cb->ac', emb_norm, emb_norm)
        W = W.fill_diagonal_(0) # official code does not has this
        W = self.clip(W)

        # init prob, official code use stratified sample, here we simply use uniform sample without stratification
        inds_shuf = torch.randperm(n).to(device)
        n_select = n_cls * self.n_lbs_per_cls
        i_onehot = inds_shuf[:n_select]
        i_prob = inds_shuf[n_select:]
        j = lbs[i_onehot]

        X = torch.zeros_like(logits)
        probs = logits.softmax(dim=1)
        X[i_onehot, j] = 1.
        X[i_prob] = probs[i_prob]

        # reinforce iteratively
        for _ in range(self.n_iters):
            X = torch.einsum('ab,bc->ac', W, X)
            Xsum = X.sum(dim=1, keepdims=True) + 1e-6
            X = X / Xsum

        # cross entropy
        X = X.log()
        loss = F.nll_loss(X, lbs, reduction='mean')
        return loss


if __name__ == '__main__':
    in_feats = 2048
    n_ids = 100
    inten = torch.randn(16, in_feats).cuda()
    lbs = torch.randint(0, n_ids, (16, )).cuda()

    # method 1: has fc inside, only need to feed emb
    crit = GroupLoss(in_feats=in_feats, n_ids=n_ids).cuda()
    loss = crit(inten, lbs)
    print(loss)
    loss.backward()

    # method 2: no fc inside, need to input logits
    model = nn.Linear(in_feats, n_ids).cuda()
    crit = GroupLoss(in_feats=in_feats, n_ids=n_ids, has_fc=False).cuda()
    logits = model(inten)
    loss = crit(inten, lbs, logits)
    print(loss)
    loss.backward()

