
import torch
import torch.nn as nn
import torch.nn.functional as F


'''
modified from the method proposed in this paper: [Context Prior for Scene Segmentation](https://arxiv.org/abs/2004.01547)

The original paper uses global pairwise affinity to compute loss, while here we simply uses pair affinity within kxk local region. Besides, the so-called global term is removed.
'''



from .one_hot import convert_to_one_hot
class AffinityLoss(nn.Module):

    def __init__(self, kernel_size=3, ignore_index=-100):
        super(AffinityLoss, self).__init__()
        self.kernel_size = kernel_size
        self.ignore_index = ignore_index
        self.unfold = nn.Unfold(kernel_size=kernel_size)
        self.bce = nn.BCEWithLogitsLoss(reduction='mean')

    def forward(self, logits, labels):
        '''
        usage similar to nn.CrossEntropyLoss:
            >>> criteria = AffinityLoss(kernel_size=3, ignore_index=255)
            >>> logits = torch.randn(8, 19, 384, 384) # nchw
            >>> lbs = torch.randint(0, 19, (8, 384, 384)) # nhw
            >>> loss = criteria(logits, lbs)
        '''
        n, c, h, w = logits.size()
        context_size = self.kernel_size * self.kernel_size
        lb_one_hot = convert_to_one_hot(labels, c, self.ignore_index).detach()
        logits_unfold = self.unfold(logits).view(n, c, context_size, -1)
        lbs_unfold = self.unfold(lb_one_hot).view(n, c, context_size, -1)
        aff_map = torch.einsum('ncal,ncbl->nabl', logits_unfold, logits_unfold)
        lb_map = torch.einsum('ncal,ncbl->nabl', lbs_unfold, lbs_unfold)
        loss = self.bce(aff_map, lb_map)
        return loss



class AffinityFieldLoss(nn.Module):
    '''
        loss proposed in the paper: https://arxiv.org/abs/1803.10335
        used for sigmentation tasks
    '''
    def __init__(self, kl_margin, lambda_edge=1., lambda_not_edge=1., ignore_lb=255):
        super(AffinityFieldLoss, self).__init__()
        self.kl_margin = kl_margin
        self.ignore_lb = ignore_lb
        self.lambda_edge = lambda_edge
        self.lambda_not_edge = lambda_not_edge
        self.kldiv = nn.KLDivLoss(reduction='none')

    def forward(self, logits, labels):
        ignore_mask = labels.cpu() == self.ignore_lb
        n_valid = ignore_mask.numel() - ignore_mask.sum().item()
        indices = [
                # center,               # edge
            ((1, None, None, None), (None, -1, None, None)), # up
            ((None, -1, None, None), (1, None, None, None)), # down
            ((None, None, 1, None), (None, None, None, -1)), # left
            ((None, None, None, -1), (None, None, 1, None)), # right
            ((1, None, 1, None), (None, -1, None, -1)), # up-left
            ((1, None, None, -1), (None, -1, 1, None)), # up-right
            ((None, -1, 1, None), (1, None, None, -1)), # down-left
            ((None, -1, None, -1), (1, None, 1, None)), # down-right
        ]

        losses = []
        probs = torch.softmax(logits, dim=1)
        log_probs = torch.log_softmax(logits, dim=1)
        for idx_c, idx_e in indices:
            lbcenter = labels[:, idx_c[0]:idx_c[1], idx_c[2]:idx_c[3]].detach()
            lbedge = labels[:, idx_e[0]:idx_e[1], idx_e[2]:idx_e[3]].detach()
            igncenter = ignore_mask[:, idx_c[0]:idx_c[1], idx_c[2]:idx_c[3]].detach()
            ignedge = ignore_mask[:, idx_e[0]:idx_e[1], idx_e[2]:idx_e[3]].detach()
            lgp_center = probs[:, :, idx_c[0]:idx_c[1], idx_c[2]:idx_c[3]]
            lgp_edge = probs[:, :, idx_e[0]:idx_e[1], idx_e[2]:idx_e[3]]
            prob_edge = probs[:, :, idx_e[0]:idx_e[1], idx_e[2]:idx_e[3]]
            kldiv = (prob_edge * (lgp_edge - lgp_center)).sum(dim=1)

            kldiv[ignedge | igncenter] = 0
            loss = torch.where(
                lbcenter == lbedge,
                self.lambda_edge * kldiv,
                self.lambda_not_edge * F.relu(self.kl_margin - kldiv, inplace=True)
            ).sum() / n_valid
            losses.append(loss)

        return sum(losses) / 8



if __name__ == '__main__':
    #  criteria = AffinityFieldLoss(kl_margin=3.)
    criteria = AffinityLoss(kernel_size=3, ignore_index=255)
    #  criteria.cuda()

    logits = torch.randn(8, 19, 768, 768).cuda().half()
    labels = torch.randint(0, 19, (8, 768, 768)).cuda()
    labels[0, 30:35, 40:45] = 255
    labels[1, 0:5, 40:45] = 255

    loss = criteria(logits, labels)

    #  scores = torch.softmax(logits, dim=1)
    #  loss = criteria(scores, labels)
    #  print(loss)
    #  loss.backward()
