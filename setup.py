
from setuptools import setup, Extension, find_packages
from torch.utils import cpp_extension

'''
    python setup.py install
    usage: import torch first, then import this module
'''

setup(
    name='pytorch_loss',
    ext_modules=[
        cpp_extension.CUDAExtension(
            'focal_cpp',
            ['csrc/focal_kernel.cu', ]),
        cpp_extension.CUDAExtension(
            'mish_cpp',
            ['csrc/mish_kernel.cu']),
        cpp_extension.CUDAExtension(
            'swish_cpp',
            ['csrc/swish_kernel.cu']),
        cpp_extension.CUDAExtension(
            'soft_dice_cpp',
            ['csrc/soft_dice_kernel_v2.cu']),
        cpp_extension.CUDAExtension(
            'lsr_cpp',
            ['csrc/lsr_kernel.cu']),
        cpp_extension.CUDAExtension(
            'large_margin_cpp',
            ['csrc/large_margin_kernel.cu']),
        cpp_extension.CUDAExtension(
            'ohem_cpp',
            ['csrc/ohem_label_kernel.cu']),
        cpp_extension.CUDAExtension(
            'one_hot_cpp',
            ['csrc/one_hot_kernel.cu']),
        cpp_extension.CUDAExtension(
            'lovasz_softmax_cpp',
            ['csrc/lovasz_softmax.cu']),
        cpp_extension.CUDAExtension(
            'taylor_softmax_cpp',
            ['csrc/taylor_softmax.cu']),
        cpp_extension.CUDAExtension(
            'layer_norm_cpp',
            ['csrc/layer_norm.cu']),
    ],
    cmdclass={'build_ext': cpp_extension.BuildExtension},
    packages=find_packages()
)
