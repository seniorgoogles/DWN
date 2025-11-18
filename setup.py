import os
import torch
import setuptools
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

cpp_dir = os.path.join('src', 'torch_dwn', 'custom_operators', 'cuda')
cuda_dir = os.path.join('src', 'torch_dwn', 'custom_operators', 'cuda')

ext_modules = []

for filename in os.listdir(cpp_dir):
    if filename.endswith('.cpp') and not filename.endswith('_kernel.cpp'):
        module_name = filename[:-4]  # Remove '.cpp' (4 characters)
        # Check if this is a CUDA extension (has a corresponding _kernel.cu file)
        kernel_filename = module_name + '_kernel.cu'
        if os.path.exists(os.path.join(cuda_dir, kernel_filename)):
            ext_modules.append(CUDAExtension(module_name, [os.path.join(cuda_dir, filename), os.path.join(cuda_dir, kernel_filename)]))
        else:
            ext_modules.append(CppExtension(module_name, [os.path.join(cpp_dir, filename)]))

setup(
    name='torch_dwn',
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension},
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    version="1.0.9",
    author="Alan T. L. Bacellar",
    author_email="alanbacellar@gmail.com",
    description="Differentiable Weightless Neural Networks (DWN) PyTorch Module",
    url="https://github.com/alanbacellar/DWN",
    install_requires=[
        'torch'
    ]
)
