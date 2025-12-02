from multiprocessing import shared_memory
from filelock import FileLock
from lightllm.utils.log_utils import init_logger
import os
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
    强制创建新的共享内存。
    Windows 极速版：如果原名被占用或处于删除等待状态，直接改名创建，不等待。
    """
    gc.collect()  # 主动 GC

    # --- 阶段 1: 尝试复用现有的 (Inspection & Reuse) ---
    try:
        existing_shm = shared_memory.SharedMemory(name=name, create=False)
        # 如果尺寸足够大，直接复用
        if existing_shm.size >= int(expected_size):
            return existing_shm
        
        # 尺寸太小，尝试销毁（Windows上这步是“尽力而为”）
        existing_shm.close()
        try:
            existing_shm.unlink()
        except:
            pass 
    except FileNotFoundError:
        pass
    except Exception as e:
        logger.warning(f"Error checking existing SHM {name}: {e}")

    # --- 阶段 2: 尝试创建 (Create or Rename) ---
    
    # 尝试 1:以此名字创建
    try:
        shm = shared_memory.SharedMemory(name=name, create=True, size=int(expected_size))
        return shm
    except Exception:
        # 忽略具体错误（无论是 FileExistsError 还是 PermissionError）
        # 只要原名创建失败，说明被占用了
        pass

    # 尝试 2: 原名被占用，生成唯一新名字
    # 格式: 原名_随机UUID前8位
    new_name = f"{name}_{uuid.uuid4().hex[:8]}"
    
    logger.warning(f"SHM name '{name}' is zombie/locked. Renaming to '{new_name}'")
    
    # 直接以新名字创建，如果这次还失败，那就是系统资源问题了，抛出异常
    try:
        shm = shared_memory.SharedMemory(name=new_name, create=True, size=int(expected_size))
        return shm
    except Exception as e:
        error_msg = f"Failed to create renamed shared memory {new_name}. Error: {e}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)


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