from __future__ import annotations

import logging
import os
import sys
import sysconfig

logger = logging.getLogger("idle_scribe.cuda")

_registered = False


def register_cuda_dll_dirs() -> None:
    """Make the pip-installed NVIDIA CUDA runtime DLLs loadable on Windows.

    faster-whisper/ctranslate2 load cuBLAS and cuDNN lazily at inference time.
    The pip packages drop those DLLs in site-packages/nvidia/*/bin, which is not
    on Windows' DLL search path — so without this the model loads but inference
    fails with "cublas64_12.dll is not found". No-op everywhere but Windows.
    """
    global _registered
    if _registered or sys.platform != "win32":
        return

    nvidia_root = os.path.join(sysconfig.get_paths()["purelib"], "nvidia")
    bin_dirs: list[str] = []
    if os.path.isdir(nvidia_root):
        for sub in os.listdir(nvidia_root):
            bin_dir = os.path.join(nvidia_root, sub, "bin")
            if os.path.isdir(bin_dir):
                bin_dirs.append(bin_dir)

    for bin_dir in bin_dirs:
        # add_dll_directory alone is not enough: ctranslate2 loads cuBLAS via a
        # path that ignores it, so also prepend to PATH (honored by LoadLibrary).
        os.add_dll_directory(bin_dir)
    if bin_dirs:
        os.environ["PATH"] = os.pathsep.join(bin_dirs + [os.environ.get("PATH", "")])
        logger.debug("Registered %d CUDA DLL dirs", len(bin_dirs))
    _registered = True
