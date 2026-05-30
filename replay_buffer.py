import numpy as np
import os
import re
from typing import List, Dict


class ReplayBuffer:
    """样本池管理类，用于存储和管理自对弈数据"""
    
    def __init__(self, max_size: int = 80000):
        """
        初始化样本池
        
        Args:
            max_size: 样本池最大容量
        """
        self.max_size = max_size
        self.buffer: List[Dict] = []
    
    def add(self, examples: List[Dict]):
        """
        添加新样本到样本池
        
        Args:
            examples: 新的样本列表
        """
        self.buffer.extend(examples)
        # 如果超过最大容量，移除最旧的样本
        if len(self.buffer) > self.max_size:
            self.buffer = self.buffer[-self.max_size:]

    def load_from_file(self, path: str):
        """从 .npz 文件加载样本并添加到样本池。

        会先清空现有样本池，然后加载文件中的所有样本。
        如果样本数超过 max_size，只保留最新的 max_size 条。

        Args:
            path: .npz 文件路径（由 self_play.save_npz 生成）
        """
        from self_play import load_npz
        examples = load_npz(path)
        self.buffer = examples
        if len(self.buffer) > self.max_size:
            self.buffer = self.buffer[-self.max_size:]
        print(f"  Loaded {len(examples)} examples from {path}, buffer size: {self.size()}")

    def load_from_folder(self, folder_path: str):
        """从训练文件夹加载所有 iteration_*_data.npz 到样本池。

        按迭代编号排序，超过 max_size 只保留最新数据。
        不会清空现有数据，而是追加。

        Args:
            folder_path: 训练 run 目录路径
        """
        if not os.path.isdir(folder_path):
            print(f"  WARNING: Folder not found: {folder_path}")
            return
        pattern = re.compile(r"iteration_(\d+)_data\.npz$")
        files = []
        for f in os.listdir(folder_path):
            m = pattern.match(f)
            if m:
                files.append((int(m.group(1)), os.path.join(folder_path, f)))
        if not files:
            print(f"  WARNING: No iteration_*_data.npz in {folder_path}")
            return
        files.sort(key=lambda x: x[0])
        from self_play import load_npz
        total = 0
        for iter_num, fp in files:
            try:
                examples = load_npz(fp)
                self.buffer.extend(examples)
                total += len(examples)
            except Exception as e:
                print(f"  WARNING: skip {fp}: {e}")
        if len(self.buffer) > self.max_size:
            self.buffer = self.buffer[-self.max_size:]
        print(f"  Loaded {total} examples from {len(files)} files in {folder_path}, buffer={self.size()}")

    def get_all(self) -> List[Dict]:
        """获取样本池中的所有样本"""
        return self.buffer
    
    def size(self) -> int:
        """获取当前样本池大小"""
        return len(self.buffer)
    
    def clear(self):
        """清空样本池"""
        self.buffer = []
