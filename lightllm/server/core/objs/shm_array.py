import sys
import numpy as np
from multiprocessing import shared_memory
from lightllm.utils.log_utils import init_logger
from lightllm.utils.shm_utils import create_or_link_shm

logger = init_logger(__name__)


class ShmArray:
    def __init__(self, name, shape, dtype):
        self.shm = None
        self.arr = None
        self.name = name
        self.dtype_byte_num = np.array([1], dtype=dtype).dtype.itemsize
        self.dest_size = np.prod(shape) * self.dtype_byte_num
        self.shape = shape
        self.dtype = dtype

    def create_shm(self):
        # 获取原本想要请求的大小
        request_size = self.dest_size
        
        # === Windows 兼容性修复 ===
        if sys.platform == 'win32':
            # 在 Windows 上，共享内存一旦创建无法轻易调整大小。
            # 为了避免因下一次请求数据稍微变大而导致 crash，
            # 我们直接申请一个足够大的固定空间 (例如 10MB)。
            # Logprobs 数据通常只有几十 KB 到几百 KB，10MB 足够容纳绝大多数情况。
            MIN_WIN_SHM_SIZE = 10 * 1024 * 1024  # 10 MB
            
            if request_size < MIN_WIN_SHM_SIZE:
                request_size = MIN_WIN_SHM_SIZE
        # ==========================

        # 使用调整后的大小去创建或连接
        self.shm = create_or_link_shm(self.name, request_size)
        self.arr = np.ndarray(self.shape, dtype=self.dtype, buffer=self.shm.buf)

    def link_shm(self):
        self.shm = create_or_link_shm(self.name, self.dest_size, force_mode="link")
        # [Windows兼容] 允许实际内存大于申请内存（因为Page对齐）
        assert self.shm.size >= self.dest_size
        self.arr = np.ndarray(self.shape, dtype=self.dtype, buffer=self.shm.buf)
        return

    def close_shm(self):
        if self.shm is not None:
            self.shm.close()
            self.shm.unlink()
            self.shm = None
            self.arr = None
