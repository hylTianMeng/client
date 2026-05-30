"""
文件级经验回放池（File-based Replay Buffer）

设计：
- 所有数据以 .npz 文件存储在 replay_data/ 文件夹
- 最多保留 max_files 个文件（默认 50），FIFO 淘汰旧文件
- 训练时随机采样 n = buffer_size / 1000 个文件加载到内存
- 兼容现有 save_npz / load_npz 格式
- 类名 ReplayBuffer 不变，保持向后兼容
"""

import numpy as np
import os
import re
import random
import time
from typing import List, Dict, Optional
from glob import glob


# ============================================================
#  默认数据文件夹
# ============================================================
_DEFAULT_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "replay_data")


def _ensure_data_dir(data_dir: str = _DEFAULT_DATA_DIR) -> str:
    """确保数据文件夹存在，返回其路径。"""
    os.makedirs(data_dir, exist_ok=True)
    return data_dir


def _list_npz_files(data_dir: str) -> List[str]:
    """列出数据文件夹中所有 .npz 文件，按文件名排序（时间戳顺序）。"""
    files = glob(os.path.join(data_dir, "*.npz"))
    files.sort()
    return files


# ============================================================
#  ReplayBuffer — 文件级经验回放池（兼容旧接口）
# ============================================================
class ReplayBuffer:
    """
    文件级经验回放池。

    核心行为：
    - add(examples) → 保存为一个新的 .npz 到 replay_data/，超过 max_files 则 FIFO 淘汰
    - sample_and_load() → 随机采样 n = buffer_size / 1000 个文件，加载到内存
    - get_all() → 返回当前内存中的样本
    - load_from_file / load_from_folder → 兼容旧接口
    """

    def __init__(self, max_size: int = 80000, max_files: int = 50,
                 data_dir: str = _DEFAULT_DATA_DIR):
        """
        Args:
            max_size: 样本池目标容量（决定采样文件数 n = max_size / 1000）
            max_files: 最多保留的 .npz 文件数量
            data_dir: 数据存储文件夹路径
        """
        self.max_size = max_size
        self.max_files = max_files
        self.data_dir = _ensure_data_dir(data_dir)

        # 内存缓存：当前已加载的样本
        self.buffer: List[Dict] = []
        # 记录当前缓存来自哪些文件
        self._loaded_files: List[str] = []

    # ------------------------------------------------------------------
    #  属性
    # ------------------------------------------------------------------

    @property
    def n_sample_files(self) -> int:
        """训练时应随机采样的文件数 = max_size / 1000（至少 1）。"""
        return max(1, self.max_size // 1000)

    # ------------------------------------------------------------------
    #  保存新数据 → .npz 文件
    # ------------------------------------------------------------------

    def add(self, examples: List[Dict], filepath: Optional[str] = None) -> str:
        """
        添加新样本：保存为 .npz 文件到 replay_data/，并加入内存缓存。
        超过 max_files 时自动 FIFO 淘汰最旧文件。

        Args:
            examples: 样本列表
            filepath: 可指定文件路径；为 None 则自动生成时间戳文件名

        Returns:
            str: 实际保存的文件路径
        """
        from self_play import save_npz

        if filepath is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
            micro = str(int((time.time() % 1) * 1_000_000)).zfill(6)
            filename = f"replay_{timestamp}_{micro}.npz"
            filepath = os.path.join(self.data_dir, filename)

        save_npz(filepath, examples)
        print(f"  [ReplayBuffer] Saved {len(examples)} examples → {os.path.basename(filepath)}")

        # 加入内存缓存
        self.buffer.extend(examples)
        # 内存中也不超过 max_size（保留最新）
        if len(self.buffer) > self.max_size:
            self.buffer = self.buffer[-self.max_size:]

        # FIFO 淘汰：超过 max_files 则删最旧文件
        self._enforce_file_limit()

        return filepath

    def _enforce_file_limit(self):
        """确保 .npz 文件数不超过 max_files，删除最旧的文件。"""
        files = _list_npz_files(self.data_dir)
        while len(files) > self.max_files:
            oldest = files[0]
            try:
                os.remove(oldest)
                print(f"  [ReplayBuffer] FIFO evict: removed {os.path.basename(oldest)}")
            except OSError as e:
                print(f"  [ReplayBuffer] WARNING: cannot remove {oldest}: {e}")
            files = _list_npz_files(self.data_dir)

    # ------------------------------------------------------------------
    #  训练采样：随机取 n 个文件 → 加载到内存
    # ------------------------------------------------------------------

    def sample_and_load(self) -> List[Dict]:
        """
        训练前调用：从 replay_data/ 随机采样 n = max_size/1000 个 .npz 文件，
        加载全部样本到内存，返回样本列表。

        Returns:
            List[Dict]: 采样得到的全部样本
        """
        all_files = _list_npz_files(self.data_dir)
        if not all_files:
            print("  [ReplayBuffer] WARNING: No .npz files in replay_data/, nothing to sample.")
            self.buffer = []
            self._loaded_files = []
            return []

        n = min(self.n_sample_files, len(all_files))
        sampled = random.sample(all_files, n)

        from self_play import load_npz

        self.buffer = []
        for fp in sampled:
            try:
                exs = load_npz(fp)
                self.buffer.extend(exs)
            except Exception as e:
                print(f"  [ReplayBuffer] WARNING: Failed to load {os.path.basename(fp)}: {e}")

        self._loaded_files = list(sampled)
        # 内存中也不超过 max_size
        if len(self.buffer) > self.max_size:
            self.buffer = self.buffer[-self.max_size:]

        print(f"  [ReplayBuffer] Sampled {n}/{len(all_files)} files → "
              f"{len(self.buffer)} examples in memory")
        return self.buffer

    # ------------------------------------------------------------------
    #  兼容旧接口：从指定文件/文件夹加载
    # ------------------------------------------------------------------

    def load_from_file(self, path: str):
        """从单个 .npz 文件加载样本到内存（兼容旧接口）。"""
        from self_play import load_npz
        examples = load_npz(path)
        self.buffer = examples
        if len(self.buffer) > self.max_size:
            self.buffer = self.buffer[-self.max_size:]
        print(f"  [ReplayBuffer] Loaded {len(examples)} examples from "
              f"{os.path.basename(path)}, buffer size: {self.size()}")

    def load_from_folder(self, folder_path: str):
        """从训练文件夹加载所有 iteration_*_data.npz 到内存（兼容旧接口）。"""
        if not os.path.isdir(folder_path):
            print(f"  [ReplayBuffer] WARNING: Folder not found: {folder_path}")
            return
        pattern = re.compile(r"iteration_(\d+)_data\.npz$")
        files = []
        for f in os.listdir(folder_path):
            m = pattern.match(f)
            if m:
                files.append((int(m.group(1)), os.path.join(folder_path, f)))
        if not files:
            print(f"  [ReplayBuffer] WARNING: No iteration_*_data.npz in {folder_path}")
            return
        files.sort(key=lambda x: x[0])
        from self_play import load_npz
        total = 0
        for _, fp in files:
            try:
                exs = load_npz(fp)
                self.buffer.extend(exs)
                total += len(exs)
            except Exception as e:
                print(f"  [ReplayBuffer] WARNING: skip {fp}: {e}")
        if len(self.buffer) > self.max_size:
            self.buffer = self.buffer[-self.max_size:]
        print(f"  [ReplayBuffer] Loaded {total} examples from {len(files)} files "
              f"in {folder_path}, buffer={self.size()}")

    # ------------------------------------------------------------------
    #  查询 & 清理
    # ------------------------------------------------------------------

    def get_all(self) -> List[Dict]:
        """获取当前内存中的所有样本。"""
        return self.buffer

    def size(self) -> int:
        """获取当前内存中样本数。"""
        return len(self.buffer)

    def clear(self):
        """清空内存中的样本（不删除磁盘文件）。"""
        self.buffer = []
        self._loaded_files = []

    # ------------------------------------------------------------------
    #  维护
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        """返回回放池统计信息。"""
        files = _list_npz_files(self.data_dir)
        total_size = sum(os.path.getsize(f) for f in files) if files else 0
        return {
            "data_dir": self.data_dir,
            "disk_files": len(files),
            "max_files": self.max_files,
            "buffer_size": self.max_size,
            "n_sample_files": self.n_sample_files,
            "mem_examples": len(self.buffer),
            "disk_size_mb": total_size / (1024 * 1024),
        }

    def disk_file_count(self) -> int:
        """磁盘上 replay_data/ 中的 .npz 文件数。"""
        return len(_list_npz_files(self.data_dir))
