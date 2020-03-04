
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self,
                 alpha=0.25,
                 gamma=2,
                 reduction='mean',):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.crit = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, label):
        '''
        args:
            logits: tensor of shape (N, ...)
            label: tensor of shape(N, ...)
        '''

        # compute loss
        logits = logits.float() # use fp32 if logits is fp16
        with torch.no_grad():
            alpha = torch.empty_like(logits).fill_(1 - self.alpha)
            alpha[label == 1] = self.alpha

        probs = torch.sigmoid(logits)
        pt = torch.where(label == 1, probs, 1 - probs)
        ce_loss = self.crit(logits, label.double())
        loss = (alpha * torch.pow(1 - pt, self.gamma) * ce_loss).sum(dim=1)
        if self.reduction == 'mean':
            loss = loss.mean()
        if self.reduction == 'sum':
            loss = loss.sum()
        return loss



if __name__ == '__main__':
    criteria1 = FocalLoss(alpha=1, gamma=2)
    criteria2 = FocalLoss(alpha=1, gamma=2)
    criteria3 = torch.nn.CrossEntropyLoss(ignore_index=255)
    logits = torch.randn(16, 19, 14, 14)
    label = torch.randint(0, 2, (16, 19, 14, 14))
    #  label[2, 3, 3] = 255
    loss = criteria1(logits, label)
    print(loss.item())
    loss = criteria2(logits, label)
    print(loss.item())
    #  print(sigmoid_focal_loss(logits, label))



    #  loss = criteria1(logits, label)
    #  print(loss.item())
    #  loss = criteria1(logits, label)
    #  print(loss.item())
    #  loss = criteria2(logits, label)
    #  print(loss.item())
