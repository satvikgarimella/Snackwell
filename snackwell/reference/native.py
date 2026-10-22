"""Optional compiled integer checker; never supplies executed model outputs."""

import ctypes
import hashlib
import os
import shutil
import subprocess
import warnings
from functools import lru_cache
from pathlib import Path

from snackwell.paths import ROOT


@lru_cache(maxsize=1)
def native_library():
    """Keep the pure-Python checker available on machines without a C++ compiler."""
    compiler = shutil.which(os.environ.get("CXX", "c++"))
    if compiler is None:
        return None
    import fcntl

    source = Path(__file__).with_name("fp32_native.cpp")
    flags = ["-std=c++17", "-O3", "-shared", "-fPIC"]
    try:
        version = subprocess.check_output([compiler, "--version"], timeout=10)
        identity = source.read_bytes() + version + str((compiler, flags)).encode()
        directory = ROOT / "build/reference" / hashlib.sha256(identity).hexdigest()[:20]
        directory.mkdir(parents=True, exist_ok=True)
        library_path = directory / "fp32.so"
        with (directory / "build.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not library_path.exists():
                temporary = directory / "fp32.tmp.so"
                result = subprocess.run(
                    [compiler, *flags, str(source), "-o", str(temporary)],
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                (directory / "compile.log").write_text(result.stdout + result.stderr)
                if result.returncode:
                    raise RuntimeError(f"native checker compilation failed; see {directory}")
                temporary.replace(library_path)
        library = ctypes.CDLL(str(library_path))
        word, pointer = ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32)
        library.snackwell_fp32.argtypes = [word, word, word]
        library.snackwell_fp32.restype = word
        library.snackwell_tensor_product.argtypes = [pointer] * 4 + [ctypes.c_int] * 4
        library.snackwell_tensor_product.restype = None
        return library
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        warnings.warn(f"Using Python integer verification: {error}", stacklevel=2)
        return None


def tensor_product(a, b, m, n, k, transpose_b=False, initial=None):
    if (
        any(type(size) is not int or size < 1 for size in (m, n, k))
        or k % 4
        or len(a) != m * k
        or len(b) != k * n
        or (initial is not None and len(initial) != m * n)
    ):
        raise ValueError("invalid native tensor dimensions")
    library = native_library()
    if library is None:
        from snackwell.reference.inference import tensor_product as python_product

        return python_product(a, b, m, n, k, transpose_b, initial)
    word = ctypes.c_uint32
    result = (word * (m * n))()
    library.snackwell_tensor_product(
        (word * len(a))(*a),
        (word * len(b))(*b),
        None if initial is None else (word * len(initial))(*initial),
        result,
        m,
        n,
        k,
        transpose_b,
    )
    return list(result)
