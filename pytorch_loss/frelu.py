

import torch
import torch.nn as nn


class FReLU(nn.Module):

    def __init__(self, in_chan):
        super(FReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan, in_chan, 3, 1, 1, groups=in_chan)
        self.bn = nn.BatchNorm2d(in_chan)
        nn.init.xavier_normal_(self.conv.weight, gain=1.)

    def forward(self, x):
        branch = self.bn(self.conv(x))
        out = torch.max(x, branch)
        return out


if __name__ == "__main__":
    m = FReLU(32)
    inten = torch.randn(4, 32, 224, 224)
    out = m(inten)
    print(out.size())
