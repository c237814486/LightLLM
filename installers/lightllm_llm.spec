# -*- mode: python ; coding: utf-8 -*-

import os
from pathlib import Path
from PyInstaller.utils.hooks import collect_all
import pathlib
import sysconfig
import sys

current_dir = Path(os.path.abspath('.'))

# 收集关键数据资源
datas = []

# 收集lightllm源码 - 仅放入 lightllm，按你的导入方式使用
lightllm_src = current_dir / 'lightllm'
if lightllm_src.exists():
    datas.append((str(lightllm_src), 'lightllm'))

include_path = pathlib.Path(sysconfig.get_paths()["include"])
if include_path.exists():
    datas.append((str(include_path), str(include_path.relative_to(sys.prefix))))

# 收集flash_attn相关文件
try:
    import flash_attn
    fa_path = pathlib.Path(flash_attn.__file__).parent
    for py_file in fa_path.rglob("*.py"):
        rel_path = py_file.relative_to(fa_path.parent)
        datas.append((str(py_file), str(rel_path.parent)))
except ImportError:
    pass

# 收集deepspeed相关文件
try:
    import deepspeed
    ds_path = pathlib.Path(deepspeed.__file__).parent
    for py_file in ds_path.rglob("*.py"):
        rel_path = py_file.relative_to(ds_path.parent)
        datas.append((str(py_file), str(rel_path.parent)))
except ImportError:
    pass

# 收集transformers相关文件
try:
    import transformers
    tf_path = pathlib.Path(transformers.__file__).parent
    # 只收集必要的文件，避免包过大
    for pattern in ["*.py", "*.json", "*.txt"]:
        for py_file in tf_path.rglob(pattern):
            if any(skip in str(py_file) for skip in ["__pycache__", ".git", "tests"]):
                continue
            rel_path = py_file.relative_to(tf_path.parent)
            datas.append((str(py_file), str(rel_path.parent)))
except ImportError:
    pass

try:
    import Cython
    cython_path = pathlib.Path(Cython.__file__).parent / "Utility"
    if cython_path.exists():
        for f in cython_path.iterdir():
            # 收集 Utility 目录下所有文件（.c/.cpp/.pxd/.h/.pyx/.py 等）
            if f.is_file():
                datas.append((str(f), "Cython/Utility"))
except ImportError:
    pass

# 直接从源码目录收集 lightllm_kernel
lightllm_kernel_src = Path('/mnt/afs/yangdeyu/dependency/LightKernel/lightllm_kernel')
_lightllm_kernel_bins = []
if lightllm_kernel_src.exists():
    # 收集 Python 源码文件
    for py_file in lightllm_kernel_src.rglob("*.py"):
        rel_path = py_file.relative_to(lightllm_kernel_src.parent)
        datas.append((str(py_file), str(rel_path.parent)))
    
    # 收集 .so 文件
    for so_file in lightllm_kernel_src.rglob("*.so"):
        _name = so_file.name
        # 核心扩展模块必须放在顶层，才能以 `import lightllm_kernel` 成功导入
        if _name.startswith('_C') and _name.endswith('.so'):
            _lightllm_kernel_bins.append((str(so_file), 'lightllm_kernel'))
        else:
            # 其余依赖性 .so 放入子目录
            rel_dir = 'lightllm_kernel/_libs'
            _lightllm_kernel_bins.append((str(so_file), rel_dir))


try:
    import triton
    triton_path = pathlib.Path(triton.__file__).parent
    for py_file in triton_path.rglob("*.py"):
        rel_path = py_file.relative_to(triton_path.parent)
        datas.append((str(py_file), str(rel_path.parent)))
except:
    pass

# 收集 vLLM 源码（便于运行期读取配置/模板等资源）
try:
    import vllm
    vllm_path = pathlib.Path(vllm.__file__).parent
    for pattern in ["*.py", "*.json", "*.txt"]:
        for f in vllm_path.rglob(pattern):
            if any(skip in str(f) for skip in ["__pycache__", ".git", "tests"]):
                continue
            rel_path = f.relative_to(vllm_path.parent)
            datas.append((str(f), str(rel_path.parent)))
except Exception:
    pass

# 收集 gunicorn 相关文件
try:
    import gunicorn
    gunicorn_path = pathlib.Path(gunicorn.__file__).parent
    for pattern in ["*.py", "*.json", "*.txt"]:
        for f in gunicorn_path.rglob(pattern):
            if any(skip in str(f) for skip in ["__pycache__", ".git", "tests"]):
                continue
            rel_path = f.relative_to(gunicorn_path.parent)
            datas.append((str(f), str(rel_path.parent)))
except Exception:
    pass

print("收集的数据文件:")
for d in datas:
    print(f"    {d}")

# 隐藏导入
hiddenimports = [
    'flash_attn',
    'flash_attn.ops.triton',
    'flash_attn.layers',
    'deepspeed',
    'deepspeed.ops.op_builder',
    'deepspeed.ops.adam',
    'deepspeed.ops.lamb',
    'deepspeed.ops.sparse_attn',
    'deepspeed.ops.transformer',
    'deepspeed.ops.adam_cpu',
    'transformers',
    'transformers.models',
    'transformers.tokenization_utils',
    'transformers.tokenization_utils_base',
    'torch',
    'torch.nn',
    'torch.nn.functional',
    'torch.distributed',
    'torch.multiprocessing',
    'numpy',
    'triton',
    'triton.language',
    'triton.ops',
    'triton.runtime',
    'triton.runtime.jit',
    'triton.compiler',
    'triton.compiler.compiler',
    'fastapi',
    'uvicorn',
    'pydantic',
    'requests',
    'aiofiles',
    'httpx',
    'websockets',
    'zmq',
    'zmq.asyncio',
    'rpyc',
    'psutil',
    'GPUtil',
    'accelerate',
    'safetensors',
    'tokenizers',
    'sentencepiece',
    'protobuf',
    'onnx',
    'onnxruntime',
    'PIL',
    'cv2',
    'librosa',
    'soundfile',
    'scipy',
    'sklearn',
    "torch.utils.cpp_extension",
    "setuptools.command.build_ext",
    "Cython.Compiler.Main",
    "easydict",
    'einops',
    'packaging',
    'regex',
    'pyzmq',
    'prometheus_client',
    'uvloop',
    'ujson',
    'frozendict',
    'atomics',
    # Web 服务器
    'gunicorn',
    'gunicorn.app',
    'gunicorn.app.wsgiapp',
    'gunicorn.workers',
    'gunicorn.workers.uvicorn_worker',
    'gunicorn.util',
    'gunicorn.config',
    # vLLM 入口
    'vllm',
    'vllm.engine',
    'vllm.worker',
    'vllm.sampling_params',
    'vllm.inputs',
    'vllm.attention',
    'vllm.attention.backends',
    'vllm.attention.ops',
    'vllm.triton_ops',
    'vllm.utils',
    'sortedcontainers',
    "lightllm-kernel",
]

# 排除不需要的模块（并避免将 lightllm 打入 PYZ，强制走磁盘源码路径）
excludes = ['lightllm']

# 收集onnxruntime
try:
    _data, _bins, _hidden = collect_all('onnxruntime')
    datas += _data
    hiddenimports += _hidden
    _onnx_bins = _bins
except Exception:
    _onnx_bins = []

# 收集triton
try:
    _data, _bins, _hidden = collect_all('triton')
    datas += _data
    hiddenimports += _hidden
    _triton_bins = _bins
except Exception:
    _triton_bins = []

# 收集 vLLM
try:
    _data, _bins, _hidden = collect_all('vllm')
    datas += _data
    hiddenimports += _hidden
    _vllm_bins = _bins
    # 额外：打包 vllm 下的 .so 扩展
    import glob as _glob
    import os as _os
    import vllm as _vllm
    _vllm_dir = _os.path.dirname(_vllm.__file__)
    for so_path in _glob.glob(_os.path.join(_vllm_dir, "**", "*.so"), recursive=True):
        rel_dir = _os.path.dirname(so_path).replace(_vllm_dir, 'vllm')
        _vllm_bins.append((so_path, rel_dir))
except Exception:
    _vllm_bins = []

# 收集 gunicorn
try:
    _data, _bins, _hidden = collect_all('gunicorn')
    datas += _data
    hiddenimports += _hidden
    _gunicorn_bins = _bins
except Exception:
    _gunicorn_bins = []

# 合并所有二进制文件
all_bins = _onnx_bins + _triton_bins + _vllm_bins + _lightllm_kernel_bins + _gunicorn_bins

python_path = sys.executable
venv_root = os.path.dirname(os.path.dirname(python_path))
lib_path = os.path.join(venv_root,"lib", f"python{sys.version_info.major}.{sys.version_info.minor}", "site-packages", "atomics", "_clib", "libpatomic.so")
all_bins.append((lib_path, 'atomics/_clib'))

a = Analysis(
    ['lightllm_launcher.py'],
    pathex=[str(current_dir)],
    binaries=all_bins,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[str(current_dir / 'pyi_runtime_hook.py')],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=True,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=None)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='lightllm_server',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='lightllm_server',
)
