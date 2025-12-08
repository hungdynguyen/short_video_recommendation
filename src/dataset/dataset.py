import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import json
from src.config import Config

class FullTrainingDataset(Dataset):
    def __init__(self, parquet_path, mappings_path):
        # 1. Load User Data
        self.data = pd.read_parquet(parquet_path)
        self.max_history = Config.MAX_HISTORY
        
        # 2. Load Mappings (để biết ID max cho Embedding Layer)
        with open(mappings_path, 'r') as f:
            self.mappings = json.load(f)
            
        # 3. Load Item Features (Pre-computed)
        print(">>> Loading Item Features into RAM...")
        
        # Load Text
        self.text_feats = np.load(Config.TEXT_FEATS_PATH).astype(np.float32)
        text_ids = np.load(Config.TEXT_IDS_PATH)
        # Map: Real_PID_String -> Index_in_Array
        self.text_map = {str(pid): i for i, pid in enumerate(text_ids)}
        
        # Load Video
        self.video_feats = np.load(Config.VIDEO_FEATS_PATH).astype(np.float32)
        video_ids = np.load(Config.VIDEO_IDS_PATH)
        self.video_map = {str(pid): i for i, pid in enumerate(video_ids)}
        
        # Mapping ngược từ ID số nguyên (trong parquet) ra Real ID (để tra cứu feature)
        # Giả sử trong mappings.json có mục 'pid': {'10023': 1, ...}
        # Ta cần đảo ngược: {1: '10023'}
        self.int2pid = {v: k for k, v in self.mappings['pid'].items()}
        
        # Normalize: remove .mp4 extension if present in mappings
        self.int2pid = {k: v.replace('.mp4', '') for k, v in self.int2pid.items()}
        
        # Vector rỗng phòng trường hợp thiếu data
        self.empty_text = np.zeros(Config.TEXT_DIM, dtype=np.float32)
        self.empty_video = np.zeros(Config.VIDEO_DIM, dtype=np.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # --- A. Xử lý User (Code của bạn) ---
        hist = row['history_pids']
        if isinstance(hist, np.ndarray): 
            hist = hist.tolist()
        # Padding History
        if len(hist) < self.max_history:
            hist = hist + [0] * (self.max_history - len(hist))
        else:
            hist = hist[-self.max_history:]
            
        # --- B. Xử lý Item (Feature Lookup) ---
        pid_int = row['pid']
        real_pid_str = self.int2pid.get(pid_int, "Unknown")
        
        # Lấy Text Vector (text_ids không có .mp4 extension)
        t_idx = self.text_map.get(real_pid_str)
        t_feat = self.text_feats[t_idx] if t_idx is not None else self.empty_text
        
        # Lấy Video Vector (video_ids cũng không có .mp4 extension)
        v_idx = self.video_map.get(real_pid_str)
        v_feat = self.video_feats[v_idx] if v_idx is not None else self.empty_video

        return {
            # User Tensors
            'user_id': torch.tensor(row['user_id'], dtype=torch.long),
            'gender': torch.tensor(row['gender'], dtype=torch.long),
            'fre_city': torch.tensor(row['fre_city'], dtype=torch.long),
            'fre_community_type': torch.tensor(row['fre_community_type'], dtype=torch.long),
            'fre_city_level': torch.tensor(row['fre_city_level'], dtype=torch.long),
            'age': torch.tensor(row['age'], dtype=torch.float),
            'mod_price': torch.tensor(row['mod_price'], dtype=torch.float),
            'p_hour': torch.tensor(row['p_hour'], dtype=torch.long),
            'p_dow': torch.tensor(row['p_dow'], dtype=torch.long),
            'history_pids': torch.tensor(hist, dtype=torch.long),
            
            # Item Tensors
            'pid': torch.tensor(pid_int, dtype=torch.long),
            'text_feat': torch.tensor(t_feat),
            'video_feat': torch.tensor(v_feat)
        }