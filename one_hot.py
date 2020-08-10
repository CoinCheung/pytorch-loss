
import torch
import torch.nn as nn
import torch.nn.functional as F



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
        with torch.no_grad():
            x = x.clone().detach()
            ignore = x == ignore_idx
            x[ignore] = 0
            out = torch.zeros(size, device=device).scatter_(1, x.unsqueeze(1), 1)
            ignore = ignore.nonzero()
            _, M = ignore.size()
            a, *b = ignore.chunk(M, dim=1)
            out[[a, torch.arange(minleng), *b]] = 0
    return out




if __name__ == "__main__":
    x = torch.randint(0, 3, (3, 4))
    print(x)
    x[1, 1] = 4
    print(x)
    out = convert_to_one_hot(x, minleng=4, ignore_idx=4)
    print(out)
