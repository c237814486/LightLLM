from multiprocessing import shared_memory
from filelock import FileLock
from lightllm.utils.log_utils import init_logger
import os
import sys
import tempfile
import platform
import time
import gc
import uuid  # 新增引用

logger = init_logger(__name__)


def create_or_link_shm(name, expected_size, force_mode=None):
    """
    Args:
        name: name of the shared memory
        expected_size: expected size of the shared memory
        force_mode: force mode
            - 'create': force create new shared memory, if exists, delete and create
            - 'link': force link to existing shared memory, if not exists, raise exception
            - None (default): smart mode, link to existing, if not exists, create

    Returns:
        shared_memory.SharedMemory: shared memory object
        
    注意：返回的 shm.name 可能与传入的 name 不同（如果发生了重命名）。
    """
    # [Windows兼容] 使用系统临时目录
    try:
        tmp_dir = tempfile.gettempdir()
        lock_name = os.path.join(tmp_dir, f"{name}.lock")
    except:
        lock_name = f"{name}.lock"

    if force_mode == "create":
        with FileLock(lock_name):
            return _force_create_shm(name, expected_size)
    elif force_mode == "link":
        return _force_link_shm(name, expected_size)
    else:
        with FileLock(lock_name):
            return _smart_create_or_link_shm(name, expected_size)


def _force_create_shm(name, expected_size):
    """
    创建共享内存。
    Linux: 强制清理旧的同名共享内存，并创建新的。
    Windows: 尝试连接现有的，如果尺寸满足则复用；如果不存在则创建；如果尺寸不足则报错。
    """
    
    # === Linux / Non-Windows 逻辑 (保持原样) ===
    if sys.platform != 'win32':
        try:
            existing_shm = shared_memory.SharedMemory(name=name)
            existing_shm.close()
            existing_shm.unlink()
        except:
            pass

        # 创建新的共享内存
        shm = shared_memory.SharedMemory(name=name, create=True, size=int(expected_size))
        return shm

    # === Windows 逻辑 ===
    else:
        try:
            # 1. 尝试连接现有的共享内存
            shm = shared_memory.SharedMemory(name=name)
            
            # 2. 如果存在，检查尺寸是否满足要求
            if shm.size >= expected_size:
                return shm
            else:
                # 尺寸不满足，关闭连接并报错
                current_size = shm.size
                shm.close()
                raise ValueError(
                    f"Shared memory '{name}' exists but size ({current_size}) "
                    f"is smaller than expected ({expected_size})."
                )
                
        except FileNotFoundError:
            # 3. 如果不存在 (FileNotFoundError)，则创建新的
            shm = shared_memory.SharedMemory(name=name, create=True, size=int(expected_size))
            return shm


def _force_link_shm(name, expected_size):
    """强制连接到已存在的共享内存"""
    try:
        shm = shared_memory.SharedMemory(name=name)
        
        # [Windows兼容] 宽松的大小检查
        if shm.size < int(expected_size):
            real_size = shm.size
            shm.close()
            raise ValueError(f"Shared memory {name} size mismatch: expected {expected_size}, got {real_size}")
        
        return shm
    except Exception as e:
        raise e


def _smart_create_or_link_shm(name, expected_size):
    """优先连接，不存在则创建"""
    try:
        shm = _force_link_shm(name=name, expected_size=expected_size)
        return shm
    except:
        pass

    return _force_create_shm(name=name, expected_size=expected_size)