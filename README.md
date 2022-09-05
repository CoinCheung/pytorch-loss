# pytorch-loss

My implementation of label-smooth, amsoftmax, partial-fc, focal-loss, dual-focal-loss, triplet-loss, giou/diou/ciou-loss/func, affinity-loss, pc_softmax_cross_entropy, ohem-loss(softmax based on line hard mining loss), large-margin-softmax(bmvc2019), lovasz-softmax-loss, and dice-loss(both generalized soft dice loss and batch soft dice loss). Maybe this is useful in my future work.


Also tried to implement swish, hard-swish(hswish) and mish activation functions.

Additionally, cuda based one-hot function is added (support label smooth).

Newly add an "Exponential Moving Average(EMA)" operator.

Add convolution ops, such as coord-conv2d, and dynamic-conv2d(dy-conv2d).

Some operators are implemented with pytorch cuda extension, so you need to compile it first: 
```
    $ python setup.py install
```

After installing, now you can pick up what you need and use the losses or ops like one of thes: 
```python
from pytorch_loss import SwishV1, SwishV2, SwishV3
from pytorch_loss import HSwishV1, HSwishV2, HSwishV3
from pytorch_loss import MishV1, MishV2, MishV3
from pytorch_loss import convert_to_one_hot, convert_to_one_hot_cu, OnehotEncoder
from pytorch_loss import EMA

from pytorch_loss import TripletLoss
from pytorch_loss import SoftDiceLossV1, SoftDiceLossV2, SoftDiceLossV3
from pytorch_loss import PCSoftmaxCrossEntropyV1, PCSoftmaxCrossEntropyV2
from pytorch_loss import LargeMarginSoftmaxV1, LargeMarginSoftmaxV2, LargeMarginSoftmaxV3
from pytorch_loss import LabelSmoothSoftmaxCEV1, LabelSmoothSoftmaxCEV2, LabelSmoothSoftmaxCEV3
from pytorch_loss import GIOULoss, DIOULoss, CIOULoss
from pytorch_loss import iou_func, giou_func, diou_func, ciou_func
from pytorch_loss import FocalLossV1, FocalLossV2, FocalLossV3
from pytorch_loss import Dual_Focal_loss
from pytorch_loss import GeneralizedSoftDiceLoss, BatchSoftDiceLoss
from pytorch_loss import AMSoftmax
from pytorch_loss import AffinityFieldLoss, AffinityLoss
from pytorch_loss import OhemCELoss, OhemLargeMarginLoss
from pytorch_loss import LovaszSoftmaxV1, LovaszSoftmaxV3
from pytorch_loss import TaylorCrossEntropyLossV1, TaylorCrossEntropyLossV3
from pytorch_loss import InfoNceDist
from pytorch_loss import PartialFCAMSoftmax

from pytorch_loss import TaylorSoftmaxV1, TaylorSoftmaxV3
from pytorch_loss import LogTaylorSoftmaxV1, LogTaylorSoftmaxV3

from pytorch_loss import CoordConv2d, DY_Conv2d
```
Note that some losses or ops have 3 versions, like `LabelSmoothSoftmaxCEV1`, `LabelSmoothSoftmaxCEV2`, `LabelSmoothSoftmaxCEV3`, here `V1` means the implementation with pure pytorch ops and use `torch.autograd` for backward computation, `V2` means implementation with pure pytorch ops but use self-derived formula for backward computation, and `V3` means implementation with cuda extension. Generally speaking, the `V3` ops are faster and more memory efficient, since I have tried to squeeze everything in one cuda kernel function, which in most cases brings less overhead than a combination of pytorch ops.


For those who happen to find this repo, if you see errors in my code, feel free to open an issue to correct me.
