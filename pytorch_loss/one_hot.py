
import torch
import torch.nn as nn
import torch.nn.functional as F



@torch.no_grad()
def convert_to_one_hot(x, minleng, ignore_idx=-1):
    '''
    encode input x into one hot
    inputs:
        x: tensor of shape (N, ...) with type long
        minleng: minimum length of one hot code, this should be larger than max value in x
        ignore_idx: the index in x that should be ignored, default is 255

    return:
        tensor of shape (N, minleng, ...) with type float
    '''
    device = x.device
    # compute output shape
    size = list(x.size())
    size.insert(1, minleng)
    assert x[x != ignore_idx].max() < minleng, "minleng should larger than max value in x"

    if ignore_idx < 0:
        out = torch.zeros(size, device=device).scatter_(1, x.unsqueeze(1), 1)
    else:
        # overcome ignore index
        x = x.clone().detach()
        ignore = x == ignore_idx
        x[ignore] = 0
        out = torch.zeros(size, device=device).scatter_(1, x.unsqueeze(1), 1)
        ignore = ignore.nonzero(as_tuple=False)
        _, M = ignore.size()
        a, *b = ignore.chunk(M, dim=1)
        out[[a, torch.arange(minleng), *b]] = 0
    return out


def convert_to_one_hot_cu(x, minleng, smooth=0., ignore_idx=-1):
    '''
    cuda version of encoding x into one hot, the difference from above is that, this support label smooth.
    inputs:
        x: tensor of shape (N, ...) with type long
        minleng: minimum length of one hot code, this should be larger than max value in x
        smooth: sets positive to **1. - smooth**, while sets negative to **smooth / minleng**
        ignore_idx: the index in x that should be ignored, default is 255

    return:
        tensor of shape (N, minleng, ...) with type float32
    '''
    import one_hot_cpp
    return one_hot_cpp.label_one_hot(x, ignore_idx, smooth, minleng)



class OnehotEncoder(nn.Module):

    def __init__(
            self,
            n_classes,
            lb_smooth=0.,
            ignore_idx=-1,
        ):
        super(OnehotEncoder, self).__init__()
        self.n_classes = n_classes
        self.lb_smooth = lb_smooth
        self.ignore_idx = ignore_idx

    @ torch.no_grad()
    def forward(self, label):
        return convert_to_one_hot_cu(
            label, self.n_classes, self.lb_smooth, self.ignore_idx).detach()


if __name__ == "__main__":
    x = torch.randint(0, 3, (3, 4))
    print(x)
    x[1, 1] = 4
    print(x)
    out = convert_to_one_hot(x, minleng=4, ignore_idx=4)
    print(out)

    x = torch.randint(0, 3, (3, 4)).cuda()
    smooth = 0.1
    out = convert_to_one_hot_cu(x, minleng=4, smooth=smooth, ignore_idx=4)
    print(out)
