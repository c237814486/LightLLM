# -*- mode: python ; coding: utf-8 -*-

import os
import sys
import sysconfig
from pathlib import Path
from PyInstaller.utils.hooks import collect_all
import pathlib


# 收集所有数据资源
datas = []

# ================== 通用资源 ==================
# Python头文件 (供Triton运行时使用)
try:
    py_include = sysconfig.get_paths().get('include') or sysconfig.get_config_var('INCLUDEPY')
    if py_include and Path(py_include).exists():
        pyver = f"python{sys.version_info.major}.{sys.version_info.minor}"
        datas.append((py_include, f"_internal/include/{pyver}"))
        # 兼容旧路径
        datas.append((py_include, str(Path(py_include).relative_to(sys.prefix))))
except Exception:
    pass


# ================== TTS相关资源 ==================
# TTS音频资源
cosyvoice_root = Path("/mnt/afs/yangdeyu/dependency/lightllm-cosyvoice-old/lightllm-cosyvoice")
assets_prompt = cosyvoice_root / 'assets' / 'prompt_audio'
if assets_prompt.exists():
    datas.append((str(assets_prompt), 'assets/prompt_audio'))

# TTS预训练模型资源
ttsfrd_res = cosyvoice_root / 'pretrained_models' / 'CosyVoice-ttsfrd' / 'resource'
if ttsfrd_res.exists():
    datas.append((str(ttsfrd_res), 'pretrained_models/CosyVoice-ttsfrd/resource'))

# light_tts源码
light_tts_src = cosyvoice_root / 'light_tts'
if light_tts_src.exists():
    datas.append((str(light_tts_src), 'light_tts'))

# cosyvoice源码
cosyvoice_src = cosyvoice_root / 'cosyvoice'
if cosyvoice_src.exists():
    datas.append((str(cosyvoice_src), 'cosyvoice'))

# third_party目录
third_party_dir = cosyvoice_root / 'third_party'
if third_party_dir.exists():
    datas.append((str(third_party_dir), 'third_party'))

# ================== LLM相关资源 ==================
# lightllm源码 (注意与light_tts的区别)
lightllm_root = Path("/mnt/afs/yangdeyu/dependency/lightllm-dev")
lightllm_src = lightllm_root / 'lightllm'
if lightllm_src.exists():
    datas.append((str(lightllm_src), 'lightllm'))

# lightllm_kernel二进制和源码
_lightllm_kernel_bins = []
lightllm_kernel_src = Path('/mnt/afs/yangdeyu/dependency/LightKernel/lightllm_kernel')
if lightllm_kernel_src.exists():
    # 收集Python源码文件
    for py_file in lightllm_kernel_src.rglob("*.py"):
        rel_path = py_file.relative_to(lightllm_kernel_src.parent)
        datas.append((str(py_file), str(rel_path.parent)))
    
    # 收集.so文件
    for so_file in lightllm_kernel_src.rglob("*.so"):
        _name = so_file.name
        if _name.startswith('_C') and _name.endswith('.so'):
            _lightllm_kernel_bins.append((str(so_file), 'lightllm_kernel'))
        else:
            rel_dir = 'lightllm_kernel/_libs'
            _lightllm_kernel_bins.append((str(so_file), rel_dir))

# ================== 依赖包源码收集 ==================
# flash_attn
try:
    import flash_attn
    fa_path = pathlib.Path(flash_attn.__file__).parent
    for py_file in fa_path.rglob("*.py"):
        rel_path = py_file.relative_to(fa_path.parent)
        datas.append((str(py_file), str(rel_path.parent)))
except ImportError:
    pass

# deepspeed
try:
    import deepspeed
    ds_path = pathlib.Path(deepspeed.__file__).parent
    for py_file in ds_path.rglob("*.py"):
        rel_path = py_file.relative_to(ds_path.parent)
        datas.append((str(py_file), str(rel_path.parent)))
except ImportError:
    pass

# transformers
try:
    import transformers
    tf_path = pathlib.Path(transformers.__file__).parent
    for pattern in ["*.py", "*.json", "*.txt"]:
        for f in tf_path.rglob(pattern):
            if any(skip in str(f) for skip in ["__pycache__", ".git", "tests"]):
                continue
            rel_path = f.relative_to(tf_path.parent)
            datas.append((str(f), str(rel_path.parent)))
except ImportError:
    pass

# xformers
try:
    import xformers
    xf_path = pathlib.Path(xformers.__file__).parent
    for pattern in ["*.py", "*.json", "*.txt"]:
        for f in xf_path.rglob(pattern):
            if any(skip in str(f) for skip in ["__pycache__", ".git", "tests"]):
                continue
            rel_path = f.relative_to(xf_path.parent)
            datas.append((str(f), str(rel_path.parent)))
except ImportError:
    pass

# Cython
try:
    import Cython
    cython_path = pathlib.Path(Cython.__file__).parent / "Utility"
    if cython_path.exists():
        for f in cython_path.iterdir():
            if f.is_file():
                datas.append((str(f), "Cython/Utility"))
except ImportError:
    pass

# triton
try:
    import triton
    triton_path = pathlib.Path(triton.__file__).parent
    for py_file in triton_path.rglob("*.py"):
        rel_path = py_file.relative_to(triton_path.parent)
        datas.append((str(py_file), str(rel_path.parent)))
except ImportError:
    pass

# vllm
try:
    import vllm
    vllm_path = pathlib.Path(vllm.__file__).parent
    for pattern in ["*.py", "*.json", "*.txt"]:
        for f in vllm_path.rglob(pattern):
            if any(skip in str(f) for skip in ["__pycache__", ".git", "tests"]):
                continue
            rel_path = f.relative_to(vllm_path.parent)
            datas.append((str(f), str(rel_path.parent)))
except ImportError:
    pass

# gunicorn
try:
    import gunicorn
    gunicorn_path = pathlib.Path(gunicorn.__file__).parent
    for pattern in ["*.py", "*.json", "*.txt", "*.so", "*.dll", "*.dylib", "*.pyd"]:
        for f in gunicorn_path.rglob(pattern):
            if any(skip in str(f) for skip in ["__pycache__", ".git", "tests"]):
                continue
            rel_path = f.relative_to(gunicorn_path.parent)
            datas.append((str(f), str(rel_path.parent)))
except ImportError:
    pass


# conformer (TTS)
try:
    import conformer
    conformer_path = pathlib.Path(conformer.__file__).parent
    for py_file in conformer_path.rglob("*.py"):
        rel_path = py_file.relative_to(conformer_path.parent)
        datas.append((str(py_file), str(rel_path.parent)))
except ImportError:
    pass

# rich
try:
    import rich
    rich_path = pathlib.Path(rich.__file__).parent
    for py_file in rich_path.rglob("*.py"):
        rel_path = py_file.relative_to(rich_path.parent)
        datas.append((str(py_file), str(rel_path.parent)))
except ImportError:
    pass

# whisper
try:
    import whisper
    whisper_path = pathlib.Path(whisper.__file__).parent
    whisper_assets = whisper_path / "assets"
    if whisper_assets.exists():
        datas.append((str(whisper_assets), "whisper/assets"))
    for npz_file in whisper_path.rglob("*.npz"):
        rel_path = npz_file.relative_to(whisper_path.parent)
        datas.append((str(npz_file), str(rel_path.parent)))
    for json_file in whisper_path.rglob("*.json"):
        rel_path = json_file.relative_to(whisper_path.parent)
        datas.append((str(json_file), str(rel_path.parent)))
except ImportError:
    pass

# ================== 隐藏导入 ==================
hiddenimports = [
    # TTS相关
    'light_tts',
    'light_tts.server.api_server',
    'cosyvoice',
    'conformer',
    'wget',
    'whisper',
    'whisper.audio',
    'whisper.model',
    'whisper.decoding',
    'whisper.tokenizer',
    'openai_whisper',
    'rich',
    
    # LLM相关
    'lightllm',
    'lightllm.server.api_server',
    'lightllm.server.api_cli',
    'lightllm.server.api_start',
    'lightllm-kernel',
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
    'gunicorn',
    'gunicorn.app',
    'gunicorn.app.wsgiapp',
    'gunicorn.workers',
    'gunicorn.workers.uvicorn_worker',
    'gunicorn.util',
    'gunicorn.config',
    
    # 通用深度学习框架
    'torch',
    'torch.nn',
    'torch.nn.functional',
    'torch.distributed',
    'torch.multiprocessing',
    'torch.utils.cpp_extension',
    'transformers',
    'transformers.models',
    'transformers.tokenization_utils',
    'transformers.tokenization_utils_base',
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
    'xformers',
    'triton',
    'triton.language',
    'triton.ops',
    'triton.runtime',
    'triton.runtime.jit',
    'triton.compiler',
    'triton.compiler.compiler',
    
    # 通用工具库
    'numpy',
    'scipy',
    'sklearn',
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
    'setuptools.command.build_ext',
    'Cython.Compiler.Main',
    'easydict',
    'einops',
    'packaging',
    'regex',
    'pyzmq',
    'prometheus_client',
    'uvloop',
    'ujson',
    'frozendict',
    'atomics',
    'tiktoken',
    'tiktoken_ext',
    'tiktoken_ext.openai_public',
    'tiktoken_ext.openai_private',
    'tiktoken_ext._tiktoken',
]

# 排除模块 - 避免重复打包
excludes = []

# ================== 收集二进制文件 ==================
all_bins = []

# 收集onnxruntime
try:
    _data, _bins, _hidden = collect_all('onnxruntime')
    datas += _data
    hiddenimports += _hidden
    all_bins += _bins
except Exception:
    pass

# 收集triton
try:
    _data, _bins, _hidden = collect_all('triton')
    datas += _data
    hiddenimports += _hidden
    all_bins += _bins
except Exception:
    pass

# 收集tiktoken
try:
    _data, _bins, _hidden = collect_all('tiktoken')
    datas += _data
    hiddenimports += _hidden + [
        'tiktoken_ext',
        'tiktoken_ext.openai_public',
        'tiktoken_ext.openai_private',
        'tiktoken_ext._tiktoken',
    ]
    all_bins += _bins
except Exception:
    pass

# 收集vllm
try:
    _data, _bins, _hidden = collect_all('vllm')
    datas += _data
    hiddenimports += _hidden
    all_bins += _bins
    # 额外收集vllm的.so文件
    import glob
    import vllm
    vllm_dir = os.path.dirname(vllm.__file__)
    for so_path in glob.glob(os.path.join(vllm_dir, "**", "*.so"), recursive=True):
        rel_dir = os.path.dirname(so_path).replace(vllm_dir, 'vllm')
        all_bins.append((so_path, rel_dir))
except Exception:
    pass

# 收集gunicorn
try:
    _data, _bins, _hidden = collect_all('gunicorn')
    datas += _data
    hiddenimports += _hidden
    all_bins += _bins
except Exception:
    pass

# 添加lightllm_kernel二进制文件
all_bins += _lightllm_kernel_bins

# 添加atomics库的动态链接库
try:
    python_path = sys.executable
    venv_root = os.path.dirname(os.path.dirname(python_path))
    lib_path = os.path.join(venv_root, "lib", f"python{sys.version_info.major}.{sys.version_info.minor}", 
                            "site-packages", "atomics", "_clib", "libpatomic.so")
    if os.path.exists(lib_path):
        all_bins.append((lib_path, 'atomics/_clib'))
except Exception:
    pass

# ================== 创建运行时钩子 ==================
# 创建运行时钩子文件（如果不存在）
runtime_hooks=[str(lightllm_root / 'pyi_runtime_hook.py')]

# ================== 构建Analysis ==================
a = Analysis(
    ['light_llm_tts_launcher.py'],  # 使用统一启动器
    pathex=[
        '/mnt/afs/yangdeyu/dependency/lightllm-cosyvoice-old/lightllm-cosyvoice',
        '/mnt/afs/yangdeyu/dependency/LightKernel', 
    ],
    binaries=all_bins,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=runtime_hooks,
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
    name='llm_tts_server',
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
    name='llm_tts_server',
)