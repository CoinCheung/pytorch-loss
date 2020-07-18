
from setuptools import setup, Extension
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
            ['csrc/focal.cpp', 'csrc/focal_kernel.cu']),
        cpp_extension.CUDAExtension(
            'mish_cpp',
            ['csrc/mish_kernel.cu']),
        cpp_extension.CUDAExtension(
            'swish_cpp',
            ['csrc/swish_kernel.cu']),
        cpp_extension.CUDAExtension(
            'soft_dice_cpp',
            ['csrc/soft_dice_kernel.cu']),
        cpp_extension.CUDAExtension(
            'lsr_cpp',
            ['csrc/lsr_kernel_v2.cu']),
        cpp_extension.CUDAExtension(
            'large_margin_cpp',
            ['csrc/large_margin_kernel.cu']),
        cpp_extension.CUDAExtension(
            'ohem_cpp',
            ['csrc/ohem_label_kernel.cu']),
    ],
    cmdclass={'build_ext': cpp_extension.BuildExtension}
)
