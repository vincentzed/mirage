from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

this_dir = os.path.dirname(os.path.abspath(__file__))

cuda_home = os.environ.get("CUDA_HOME", "/usr/local/cuda")

cuda_library_dirs = [
    os.path.join(cuda_home, "lib"),
    os.path.join(cuda_home, "lib64"),
    os.path.join(cuda_home, "lib64", "stubs"),
]

macros = [("MIRAGE_BACKEND_USE_CUDA", None), ("MIRAGE_FINGERPRINT_USE_CUDA", None)]

setup(
    name='runtime_kernel_lfm2_conv',
    ext_modules=[
        CUDAExtension(
            name='runtime_kernel_lfm2_conv',
            sources=[
                os.path.join(this_dir, 'runtime_kernel_wrapper_sm100.cu'),
            ],
            depends=[
                os.path.join(this_dir, '../../../../include/mirage/persistent_kernel/tasks/ampere/lfm2_conv.cuh'),
            ],
            define_macros=macros,
            include_dirs=[
                os.path.join(this_dir, '../../../../include/mirage/persistent_kernel/'),
                os.path.join(this_dir, '../../../../include/mirage/persistent_kernel/tasks'),
                os.path.join(this_dir, '../../../../include'),
                os.path.join(this_dir, '../../../../deps/cutlass/include'),
                os.path.join(this_dir, '../../../../deps/cutlass/tools/util/include'),
            ],
            libraries=["cuda"],
            library_dirs=cuda_library_dirs,
            extra_compile_args={
                'cxx': ['-DMIRAGE_GRACE_BLACKWELL'],
                'nvcc': [
                    '-O3',
                    # -arch=native keeps this test working across Blackwell
                    # variants (sm_100 B200, sm_103 B300); the kernel is plain
                    # SIMT so no arch-specific features are involved.
                    '-arch=native',
                    '-DMIRAGE_GRACE_BLACKWELL',
                    '-DMPK_ENABLE_TMA',
                ]
            }
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
