import sys
import time
import torch
from io import BytesIO
from multiprocessing import shared_memory


def tensor2bytes(t: torch.Tensor):
    # t = t.cpu().numpy().tobytes()
    # return t
    buf = BytesIO()
    t = t.detach().cpu()
    # 这个地方进行新的empty并复制是因为，torch的tensor save的机制存在问题
    # 如果 t 是从一个大 tensor 上切片复制下来的的tensor， 在save的时候，其
    # 会保存大tensor的所有数据，所以会导致存储开销较大，需要申请一个新的tensor
    # 并进行复制，来打断这种联系。
    dest = torch.empty_like(t)
    dest.copy_(t)
    torch.save(dest, buf, _use_new_zipfile_serialization=False, pickle_protocol=4)
    buf.seek(0)
    return buf.read()


def bytes2tensor(b):
    # return torch.from_numpy(np.frombuffer(b, dtype=np.float16)).cuda()
    return torch.load(BytesIO(b), weights_only=False)

# ========================================================
# Windows 补丁 V2：
# 1. 防止共享内存过早被回收 (Keep Alive)
# 2. 防止重复创建同名内存报错 (FileExistsError Fix)
# ========================================================
_WIN32_SHM_KEEP_ALIVE = {}

def _win32_cleanup_old_shm():
    """
    清理超过 60 秒的旧共享内存对象
    """
    now = time.time()
    expired_names = []
    
    for name, (shm, ts) in _WIN32_SHM_KEEP_ALIVE.items():
        if now - ts > 60: 
            expired_names.append(name)
    
    for name in expired_names:
        try:
            shm_obj, _ = _WIN32_SHM_KEEP_ALIVE.pop(name)
            shm_obj.close()
        except:
            pass

def create_shm(name, data):
    """
    创建共享内存并写入数据 (Windows 增强版)
    """
    size = len(data)

    if sys.platform == 'win32':
        # 1. 检查是否已经在保活列表中
        if name in _WIN32_SHM_KEEP_ALIVE:
            # 已经有了，直接更新时间戳，认为数据是一样的（基于哈希命名），直接返回
            shm, _ = _WIN32_SHM_KEEP_ALIVE[name]
            _WIN32_SHM_KEEP_ALIVE[name] = (shm, time.time())
            return

        # 2. 尝试创建
        try:
            shm = shared_memory.SharedMemory(name=name, create=True, size=size)
            # 只有新建的时候才需要写入数据
            shm.buf[:size] = data
        except FileExistsError:
            # 3. 如果报错存在，说明 OS 里有，但 Python 字典里没有（可能是残留或重启导致）
            # 这时候改为“连接”模式
            try:
                shm = shared_memory.SharedMemory(name=name, create=False)
                # 为了保险，覆盖写入一次数据（防止上次创建了但没写完）
                if shm.size >= size:
                    shm.buf[:size] = data
            except Exception as e:
                print(f"[Error] Failed to connect to existing SHM {name}: {e}")
                raise e

        # 4. 加入保活列表
        _win32_cleanup_old_shm()
        _WIN32_SHM_KEEP_ALIVE[name] = (shm, time.time())

    else:
        # === Linux 原生逻辑 ===
        # Linux 下如果有同名残留，通常 unlink 再创建比较安全，或者直接 create=True 会报错
        # 这里保留最简单的处理，假设 LightLLM 自身逻辑已处理 unlink
        try:
            shm = shared_memory.SharedMemory(name=name, create=True, size=size)
        except FileExistsError:
            # 如果 Linux 下也遇到已存在，尝试先删除旧的再创建
            try:
                temp = shared_memory.SharedMemory(name=name)
                temp.unlink()
            except:
                pass
            shm = shared_memory.SharedMemory(name=name, create=True, size=size)
            
        shm.buf[:size] = data
        shm.close()

def read_shm(name):
    """
    读取共享内存
    """
    try:
        shm = shared_memory.SharedMemory(name=name)
        data = bytes(shm.buf)
        shm.close()
        return data
    except FileNotFoundError:
        if sys.platform == 'win32':
            # 这是一个常见 Debug 信息，如果频繁出现说明 KeepAlive 时间太短
            print(f"[Warn] SHM {name} not found in Windows read_shm.")
        raise

def free_shm(name):
    """
    释放共享内存
    """
    if sys.platform == 'win32':
        # Windows: 从字典移除并 close，让系统回收
        if name in _WIN32_SHM_KEEP_ALIVE:
            try:
                shm_obj, _ = _WIN32_SHM_KEEP_ALIVE.pop(name)
                shm_obj.close()
            except:
                pass
    else:
        # Linux: Unlink
        try:
            shm = shared_memory.SharedMemory(name=name)
            shm.close()
            shm.unlink()
        except:
            pass


def get_shm_name_data(uid):
    return str(uid) + "-data"


def get_shm_name_embed(uid):
    return str(uid) + "-embed"
