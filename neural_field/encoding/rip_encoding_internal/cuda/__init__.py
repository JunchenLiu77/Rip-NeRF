import glob
import os
import shutil

from rich.console import Console
from torch.utils.cpp_extension import _get_build_directory, load

PATH = os.path.dirname(os.path.abspath(__file__))


name = "tri_encoding_cuda"
build_dir = _get_build_directory(name, verbose=False)
extra_include_paths = []
extra_cflags = ["-O3"]
extra_cuda_cflags = ["-O3"]

_C = None
sources = list(glob.glob(os.path.join(PATH, "csrc/*.cu"))) + list(
    glob.glob(os.path.join(PATH, "csrc/*.cpp"))
)

if os.listdir(build_dir) != []:
    # If the build exists, we assume the extension has been built
    # and we can load it.

    _C = load(
        name=name,
        sources=sources,
        extra_cflags=extra_cflags,
        extra_cuda_cflags=extra_cuda_cflags,
        extra_include_paths=extra_include_paths,
        verbose=True,
    )
else:
    # Build from scratch. Remove the build directory just to be safe: pytorch jit might stuck
    # if the build directory exists.
    shutil.rmtree(build_dir)
    with Console().status(
        "[bold yellow]TriEncoding: Setting up CUDA (This may take a few minutes the first time)",
        spinner="bouncingBall",
    ):
        _C = load(
            name=name,
            sources=sources,
            extra_cflags=extra_cflags,
            extra_cuda_cflags=extra_cuda_cflags,
            extra_include_paths=extra_include_paths,
            verbose=True,
        )


__all__ = ["_C"]