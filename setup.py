import sys

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension


cxx_flags = ["/std:c++20", "/EHsc"] if sys.platform == "win32" else ["-std=c++20"]

setup(
    packages=["bitsnn_cpu"],
    package_dir={"": "src"},
    ext_modules=[
        CppExtension(
            name="bitsnn_cpu._C",
            sources=[
                "csrc/bit_kernels.cpp",
                "csrc/avx2_kernels.cpp",
                "csrc/torch_ops.cpp",
            ],
            extra_compile_args={"cxx": cxx_flags},
        )
    ],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)},
)
