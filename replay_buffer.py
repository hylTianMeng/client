import numpy as np
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
    
    def get_all(self) -> List[Dict]:
        """获取样本池中的所有样本"""
        return self.buffer
    
    def size(self) -> int:
        """获取当前样本池大小"""
        return len(self.buffer)
    
    def clear(self):
        """清空样本池"""
        self.buffer = []
