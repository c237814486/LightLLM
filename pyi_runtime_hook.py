"""
PyInstaller 运行时钩子
处理冻结环境下的模块导入和环境配置
"""

import os
import sys
import warnings
import sysconfig
from pathlib import Path

# 禁用警告
warnings.filterwarnings("ignore")
inc = sysconfig.get_paths().get("include")
if inc and os.path.exists(inc):
    os.environ["CPATH"] = inc + (":" + os.environ["CPATH"] if "CPATH" in os.environ else "")


def _setup_environment():
    """设置运行时环境"""

    # 禁用typeguard
    os.environ.setdefault("TYPEGUARD_DISABLE", "true")
    try:
        import typeguard  # type: ignore

        def _no_typechecked(*args, **kwargs):
            def _decorator(func):
                return func

            return _decorator

        # Patch top-level symbol
        try:
            typeguard.typechecked = _no_typechecked  # type: ignore[attr-defined]
        except Exception:
            pass

        # Patch v4 decorators module if present
        try:
            from typeguard import decorators as _tg_decorators  # type: ignore

            _tg_decorators.typechecked = _no_typechecked  # type: ignore[attr-defined]
        except Exception:
            pass
    except Exception:
        # typeguard not installed or failed to import; nothing to do
        pass

    # 设置CUDA相关环境变量
    os.environ.setdefault("TORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

    # 设置多线程相关
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")

    # 禁用一些可能导致问题的优化
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

    # 设置triton相关 - 允许JIT并指定缓存目录，确保可获取源码
    os.environ.setdefault("TRITON_DISABLE_JIT", "0")
    os.environ.setdefault("TORCHINDUCTOR_MAX_AUTOTUNE", "0")
    os.environ.setdefault("TRITON_SKIP_AUTOTUNE", "0")
    os.environ.setdefault("TRITON_COMPILE_DISABLED", "0")

    # 设置PyTorch相关
    os.environ.setdefault("TORCH_USE_CUDA_DSA", "1")
    os.environ.setdefault("TORCH_CUDNN_V8_API_ENABLED", "1")

    # 禁用源码检查相关功能
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    os.environ.setdefault("PYTHONNOUSERSITE", "1")

    inc = sysconfig.get_paths().get("include")
    if inc and os.path.exists(inc):
        os.environ["CPATH"] = inc + (":" + os.environ["CPATH"] if "CPATH" in os.environ else "")

    # Prepare writable cache directories for Triton, TorchInductor, etc.
    cache_root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "llm_tts_server"
    triton_cache = cache_root / "triton"
    torchinductor_cache = cache_root / "torchinductor"
    triton_cache.mkdir(parents=True, exist_ok=True)
    torchinductor_cache.mkdir(parents=True, exist_ok=True)

    # Point Triton and Torch to writable caches
    os.environ.setdefault("TRITON_CACHE_DIR", str(triton_cache))
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(torchinductor_cache))
    # Avoid attempting to write inside bundled stdlib
    os.environ.setdefault("PYTORCH_KERNEL_CACHE_PATH", str(torchinductor_cache))

    # Ensure temp dir exists and is writable (for onefile extractions)
    os.environ.setdefault("TMPDIR", str(cache_root / "tmp"))
    Path(os.environ["TMPDIR"]).mkdir(parents=True, exist_ok=True)

    # Optional: verbose to help diagnose in user logs
    os.environ.setdefault("TRITON_DEBUG", "0")


def _setup_path():
    """设置Python路径"""
    if getattr(sys, "frozen", False):
        if hasattr(sys, "_MEIPASS"):
            # ONEFILE模式
            base_dir = sys._MEIPASS
        else:
            # ONEDIR模式
            base_dir = os.path.dirname(os.path.abspath(sys.argv[0]))

        # 仅添加 lightllm 到路径
        for pkg in ["lightllm", "light_tts"]:
            pkg_path = os.path.join(base_dir, pkg)
            if os.path.exists(pkg_path) and pkg_path not in sys.path:
                sys.path.insert(0, pkg_path)

        # 添加其他必要的路径
        for subdir in ["demos", "docs"]:
            subdir_path = os.path.join(base_dir, subdir)
            if os.path.exists(subdir_path) and subdir_path not in sys.path:
                sys.path.insert(0, subdir_path)


def _patch_imports():
    """修补导入问题 - 简化版本，避免递归"""
    # 确保triton缓存目录存在，避免JIT写入失败
    if getattr(sys, "frozen", False):
        base_dir = sys._MEIPASS if hasattr(sys, "_MEIPASS") else os.path.dirname(os.path.abspath(sys.argv[0]))
        cache_dir = os.path.join(base_dir, ".triton_cache")
        try:
            os.makedirs(cache_dir, exist_ok=True)
        except Exception:
            pass
        os.environ.setdefault("TRITON_CACHE_DIR", cache_dir)


def _setup_logging():
    """设置日志"""
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


# 执行设置
_setup_environment()
_setup_path()
_patch_imports()
_setup_logging()
