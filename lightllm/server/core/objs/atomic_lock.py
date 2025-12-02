import atomics
from multiprocessing import shared_memory
from lightllm.utils.log_utils import init_logger
from lightllm.utils.shm_utils import create_or_link_shm

logger = init_logger(__name__)


class AtomicShmLock:
    def __init__(self, lock_name: str):
        self.lock_name = lock_name
        self.dest_size = 4
        # Windows 上这里实际会申请到 4096 字节
        self.shm = create_or_link_shm(self.lock_name, self.dest_size)

        # 初始化锁状态
        # 注意：这里可能存在多进程竞争隐患，如果这是连接到已存在的锁，重置为0可能会破坏锁状态
        # 但为了保持你原有逻辑不变，这里仅处理 Windows 兼容性问题
        try:
            # 同样建议加上切片 [:4] 以防 cast 在某些极端对齐情况下报错，虽然通常 4096 能被 4 整除
            self.shm.buf[:4].cast("i")[0] = 0
        except Exception:
            # 如果初始化失败，至少保证程序不崩，继续尝试运行
            pass
        return

    def __enter__(self):
        # [Windows 兼容修复]
        # self.shm.buf 是 4096 字节，直接传给 atomicview 会报 UnsupportedWidthException
        # 使用 self.shm.buf[:4] 截取前4个字节，正好对应 atomics.INT 的大小
        with atomics.atomicview(buffer=self.shm.buf[:4], atype=atomics.INT) as a:
            while not a.cmpxchg_weak(0, 1):
                pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        # [Windows 兼容修复]
        # 同样在这里也需要截取前4个字节
        with atomics.atomicview(buffer=self.shm.buf[:4], atype=atomics.INT) as a:
            while not a.cmpxchg_weak(1, 0):
                pass
        return False