import torch
import json
import logging
from pathlib import Path
from typing import Dict, Union

class Normalizer:
    def __init__(self, stats_path: Union[str, Path], device: str = "cpu"):
        self.device = device
        self.stats = self._load_stats(stats_path)
        
    def _load_stats(self, path: Union[str, Path]) -> Dict[str, Dict[str, torch.Tensor]]:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Stats file not found at: {path}")
            
        with open(path, "r") as f:
            raw_stats = json.load(f)
            
        # 将 stats 转换为 Tensor 并移至 device
        stats_dict = {}
        for key, value in raw_stats.items():
            # LeRobot 的 stats 格式通常是 {key: {mean: [...], std: [...], min: [...], max: [...]}}
            # 我们主要用 mean/std 或 min/max
            stats_dict[key] = {}
            for stat_name, stat_val in value.items():
                if isinstance(stat_val, list):
                    stats_dict[key][stat_name] = torch.tensor(stat_val, dtype=torch.float32, device=self.device)
                else:
                     stats_dict[key][stat_name] = torch.tensor([stat_val], dtype=torch.float32, device=self.device)
                     
        return stats_dict

    def normalize(self, data: torch.Tensor, key: str, mode: str = "mean_std") -> torch.Tensor:
        """将真实数据 (Real World) -> 模型输入 (Model Input)"""
        if key not in self.stats:
            # 如果没有统计数据，可能是图像或其他不需要归一化的数据，直接返回
            return data
            
        stats = self.stats[key]
        
        if mode == "mean_std":
            # (x - mean) / std
            return (data - stats["mean"]) / (stats["std"] + 1e-8) # 防止除零
        elif mode == "min_max":
            # 2 * (x - min) / (max - min) - 1  -> 映射到 [-1, 1]
            return 2 * (data - stats["min"]) / (stats["max"] - stats["min"] + 1e-8) - 1
        else:
            raise ValueError(f"Unknown normalization mode: {mode}")

    def denormalize(self, data: torch.Tensor, key: str, mode: str = "mean_std") -> torch.Tensor:
        """模型输出 (Model Output) -> 真实数据 (Real World)"""
        if key not in self.stats:
            return data
            
        stats = self.stats[key]
        
        if mode == "mean_std":
            # x * std + mean
            return data * stats["std"] + stats["mean"]
        elif mode == "min_max":
            # (x + 1) / 2 * (max - min) + min
            return (data + 1) / 2 * (stats["max"] - stats["min"]) + stats["min"]
        else:
            raise ValueError(f"Unknown normalization mode: {mode}")

    def to(self, device):
        self.device = device
        for key in self.stats:
            for stat_name in self.stats[key]:
                self.stats[key][stat_name] = self.stats[key][stat_name].to(device)
        return self