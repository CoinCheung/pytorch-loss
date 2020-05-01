
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
            ['csrc/focal.cpp', 'csrc/focal_kernel.cu'])
    ],
    cmdclass={'build_ext': cpp_extension.BuildExtension}
)
