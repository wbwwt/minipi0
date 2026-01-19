# import torch
# from torch.utils.data import Dataset
# import pandas as pd
# from pathlib import Path
# import numpy as np
# import logging
# from typing import Optional

# # 替换 decord 为 opencv
# try:
#     import cv2
# except ImportError:
#     raise ImportError("请安装 opencv: pip install opencv-python-headless")

# from transformers import PreTrainedTokenizer
# from utils.normalization import Normalizer

# class Pi0Dataset(Dataset):
#     def __init__(
#         self, 
#         root_dir: str, 
#         tokenizer: PreTrainedTokenizer,
#         normalizer: Optional[Normalizer] = None,
#         split: str = "train",
#         image_size: int = 224,
#         max_token_len: int = 48,
#         action_chunk_size: int = 50,
#         video_key: str = "observation.images.cam_high",
#     ):
#         self.root_dir = Path(root_dir)
#         self.tokenizer = tokenizer
#         self.normalizer = normalizer
#         self.max_token_len = max_token_len
#         self.action_chunk_size = action_chunk_size
#         self.video_key = video_key
#         self.image_size = image_size
        
#         logging.info(f"Dataset root: {self.root_dir}")

#         # -----------------------------------------------------------
#         # 1. 加载元数据
#         # -----------------------------------------------------------
#         meta_dir = self.root_dir / "meta/episodes"
#         if not meta_dir.exists():
#             raise FileNotFoundError(f"找不到 meta/episodes 目录: {meta_dir}")
            
#         parquet_files = sorted(list(meta_dir.rglob("*.parquet")))
#         if not parquet_files:
#             raise FileNotFoundError(f"meta/episodes 下为空！")

#         print(f"✅ 发现 {len(parquet_files)} 个元数据文件")
        
#         dfs = [pd.read_parquet(p) for p in parquet_files]
#         self.episodes = pd.concat(dfs, ignore_index=True)
        
#         # 保护逻辑：如果只有一个 Episode，全用
#         total_episodes = len(self.episodes)
#         if total_episodes == 1:
#             pass # 不切分
#         else:
#             train_len = int(total_episodes * 0.95)
#             if train_len == 0 and total_episodes > 0: train_len = 1
            
#             if split == "train":
#                 self.episodes = self.episodes.iloc[:train_len]
#             else:
#                 self.episodes = self.episodes.iloc[train_len:]

#         # -----------------------------------------------------------
#         # 2. 构建索引
#         # -----------------------------------------------------------
#         self.indices = []
#         for _, row in self.episodes.iterrows():
#             ep_idx = row.get("episode_index")
#             length = row.get("length")
#             chunk_idx = row.get("chunk_index", 0) 
            
#             for frame_idx in range(length):
#                 self.indices.append({
#                     "episode_id": ep_idx,
#                     "frame_idx": frame_idx, 
#                     "chunk_index": chunk_idx
#                 })

#         self.data_cache = {} 
#         print(f"✅ 数据集加载完毕: {split} 集包含 {len(self.indices)} 帧")

#     def _get_data_chunk(self, chunk_idx):
#         if chunk_idx not in self.data_cache:
#             chunk_name = f"chunk-{chunk_idx:03d}"
#             file_name = f"file-{chunk_idx:03d}.parquet"
#             path = self.root_dir / "data" / chunk_name / file_name
            
#             if not path.exists():
#                 candidates = list((self.root_dir / "data").rglob(f"*{chunk_idx}*.parquet"))
#                 if candidates: path = candidates[0]
#                 else: raise FileNotFoundError(f"找不到数据文件: {path}")
            
#             self.data_cache[chunk_idx] = pd.read_parquet(path)
#         return self.data_cache[chunk_idx]

#     def __len__(self):
#         return len(self.indices)

#     def __getitem__(self, idx):
#         item = self.indices[idx]
#         ep_id = item['episode_id']
#         frame_idx = item['frame_idx']
#         chunk_idx = item['chunk_index']
        
#         # --- A. 获取 Data ---
#         df = self._get_data_chunk(chunk_idx)
#         episode_data = df[df["episode_index"] == ep_id]
        
#         if len(episode_data) == 0: return self._empty_sample()
#         if frame_idx >= len(episode_data): frame_idx = len(episode_data) - 1
            
#         current_data = episode_data.iloc[frame_idx]
        
#         # --- B. 获取 Video (OpenCV 版本) ---
#         chunk_str = f"chunk-{chunk_idx:03d}"
#         file_str = f"file-{chunk_idx:03d}.mp4"
#         video_path = self.root_dir / "videos" / self.video_key / chunk_str / file_str
        
#         if not video_path.exists():
#              video_path = self.root_dir / "videos" / chunk_str / file_str
        
#         if video_path.exists():
#             # OpenCV 读取逻辑
#             cap = cv2.VideoCapture(str(video_path))
            
#             # 计算绝对帧号 (对应 DataFrame 的 Index)
#             # 假设 DataFrame 的 index 与视频帧严格对应
#             abs_frame_idx = episode_data.index[frame_idx]
            
#             # ⚠️ 注意: 这里的 index 必须是 chunk 内的相对位置
#             # 如果 df = read_parquet(chunk_file)，它的 index 默认是从 0 开始的
#             # 那么直接用 abs_frame_idx 就是对的
#             row_in_chunk = df.index.get_loc(abs_frame_idx)
            
#             # 跳转到指定帧
#             cap.set(cv2.CAP_PROP_POS_FRAMES, row_in_chunk)
#             ret, frame_bgr = cap.read()
#             cap.release() # 及时释放，避免文件句柄耗尽
            
#             if ret:
#                 # BGR -> RGB
#                 frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
#                 frame_torch = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0
#                 frame_torch = torch.nn.functional.interpolate(frame_torch.unsqueeze(0), size=(self.image_size, self.image_size), mode='bilinear').squeeze(0)
#             else:
#                 print(f"⚠️ 视频读取失败 (Frame {row_in_chunk}): {video_path}")
#                 frame_torch = torch.zeros(3, self.image_size, self.image_size)
#         else:
#             frame_torch = torch.zeros(3, self.image_size, self.image_size)

#         # --- C. Text & State ---
#         task_text = current_data["task"] if "task" in current_data else "Pick up task"
#         tokens = self.tokenizer(task_text, return_tensors="pt", padding="max_length", truncation=True, max_length=self.max_token_len)

#         state_list = current_data["observation.state"]
#         if hasattr(state_list, 'tolist'): state_list = state_list.tolist()
#         state = torch.tensor(state_list, dtype=torch.float32)
#         if self.normalizer: state = self.normalizer.normalize(state, key="observation.state")

#         # --- D. Action ---
#         actions_chunk_df = episode_data.iloc[frame_idx : frame_idx + self.action_chunk_size]["action"]
#         actions_list = [a.tolist() if hasattr(a, 'tolist') else a for a in actions_chunk_df.values]
#         actions = torch.tensor(np.array(actions_list), dtype=torch.float32)

#         if actions.shape[0] < self.action_chunk_size:
#             pad_len = self.action_chunk_size - actions.shape[0]
#             if actions.shape[0] > 0:
#                 last = actions[-1].unsqueeze(0)
#                 actions = torch.cat([actions, last.repeat(pad_len, 1)], dim=0)
#             else: 
#                 actions = torch.zeros(self.action_chunk_size, 7) 

#         if self.normalizer: actions = self.normalizer.normalize(actions, key="action")

#         return {
#             "observation.images.base_0_rgb": frame_torch,
#             "observation.state": state,
#             "observation.language_instruction.input_ids": tokens["input_ids"].squeeze(0),
#             "observation.language_instruction.attention_mask": tokens["attention_mask"].squeeze(0),
#             "action": actions
#         }

#     def _empty_sample(self):
#         return {
#             "observation.images.base_0_rgb": torch.zeros(3, self.image_size, self.image_size),
#             "observation.state": torch.zeros(7),
#             "observation.language_instruction.input_ids": torch.zeros(self.max_token_len, dtype=torch.long),
#             "observation.language_instruction.attention_mask": torch.zeros(self.max_token_len, dtype=torch.long),
#             "action": torch.zeros(self.action_chunk_size, 7)
#         }

import torch
from torch.utils.data import Dataset
import logging
from pathlib import Path
from transformers import PreTrainedTokenizer
from typing import Optional

# 引入官方读取器
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from utils.normalization import Normalizer

class Pi0Dataset(Dataset):
    def __init__(
        self, 
        root_dir: str, 
        tokenizer: PreTrainedTokenizer,
        normalizer: Optional[Normalizer] = None,
        split: str = "train",
        image_size: int = 224,
        max_token_len: int = 48,
        action_chunk_size: int = 50,
        fps: int = 30,
        video_key: str = "observation.images.cam_high",
    ):
        self.root_dir = Path(root_dir)
        self.tokenizer = tokenizer
        self.normalizer = normalizer
        self.max_token_len = max_token_len
        self.image_size = image_size
        self.action_chunk_size = action_chunk_size
        self.video_key = video_key
        
        logging.info(f"Dataset root: {self.root_dir}")

        # -----------------------------------------------------------
        # 1. 计算 Split
        # -----------------------------------------------------------
        temp_dataset = LeRobotDataset(root=self.root_dir, repo_id="dummy_id")
        total_episodes = temp_dataset.num_episodes
        del temp_dataset 
        
        all_indices = list(range(total_episodes))
        train_len = int(total_episodes * 0.95)
        if train_len == 0 and total_episodes > 0: train_len = 1
        
        if split == "train":
            selected_episodes = all_indices[:train_len]
        else:
            selected_episodes = all_indices[train_len:] if train_len < total_episodes else all_indices
            
        print(f"📊 数据集划分 ({split}): 选中 {len(selected_episodes)} / {total_episodes} 个 Episodes")

        # -----------------------------------------------------------
        # 2. 初始化 LeRobotDataset
        # -----------------------------------------------------------
        dt = 1.0 / fps
        self.delta_timestamps = {
            "action": [i * dt for i in range(action_chunk_size)]
        }

        self.dataset = LeRobotDataset(
            repo_id="local_dataset",
            root=self.root_dir,
            episodes=selected_episodes,
            delta_timestamps=self.delta_timestamps,
            tolerance_s=2 * dt,
            video_backend="pyav" 
        )
        
        # -----------------------------------------------------------
        # 3. 安全获取 Stats (这里就是修复报错的地方 🛡️)
        # -----------------------------------------------------------
        self.stats = {}
        try:
            # 方案 A: 尝试从 meta 中获取 (新版 LeRobot)
            if hasattr(self.dataset, "meta") and hasattr(self.dataset.meta, "stats"):
                self.stats = self.dataset.meta.stats
            # 方案 B: 尝试直接获取 (旧版 LeRobot)
            elif hasattr(self.dataset, "stats"):
                self.stats = self.dataset.stats
        except Exception as e:
            print(f"⚠️ 警告: 无法从 dataset 对象中直接读取 stats ({e})，但这不影响训练，因为我们有 Normalizer。")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # 1. 获取数据
        item = self.dataset[idx]
        
        # 2. 图片处理
        raw_image = item[self.video_key]
        if raw_image.shape[1] != self.image_size or raw_image.shape[2] != self.image_size:
            image = torch.nn.functional.interpolate(
                raw_image.unsqueeze(0), 
                size=(self.image_size, self.image_size), 
                mode='bilinear', 
                align_corners=False
            ).squeeze(0)
        else:
            image = raw_image

        # 3. 文本处理
        task_text = item.get("task", "Do the task")
        text_tokens = self.tokenizer(
            task_text, 
            return_tensors="pt", 
            padding="max_length", 
            truncation=True, 
            max_length=self.max_token_len
        )

        # 4. 归一化
        state = item["observation.state"]
        actions = item["action"]
        
        if self.normalizer:
            state = self.normalizer.normalize(state, key="observation.state")
            actions = self.normalizer.normalize(actions, key="action")

        return {
            "observation.images.base_0_rgb": image,
            "observation.state": state,
            "observation.language_instruction.input_ids": text_tokens["input_ids"].squeeze(0),
            "observation.language_instruction.attention_mask": text_tokens["attention_mask"].squeeze(0),
            "action": actions
        }