"""Pure Content-Based Dataset - Load history content features instead of IDs."""
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
        
        print(f">>> Loaded {len(self.text_feats)} text features and {len(self.video_feats)} video features")

    def _get_content_features(self, pid_int):
        """Get text and video features for a given video ID."""
        real_pid_str = self.int2pid.get(pid_int, "Unknown")
        
        # Lấy Text Vector
        t_idx = self.text_map.get(real_pid_str)
        t_feat = self.text_feats[t_idx] if t_idx is not None else self.empty_text
        
        # Lấy Video Vector
        v_idx = self.video_map.get(real_pid_str)
        v_feat = self.video_feats[v_idx] if v_idx is not None else self.empty_video
        
        return t_feat, v_feat

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # --- A. Xử lý History PIDs ---
        hist = row['history_pids']
        if isinstance(hist, np.ndarray): 
            hist = hist.tolist()
        elif hist is None or (isinstance(hist, float) and np.isnan(hist)):
            # Handle NaN or None case (for cold-start users with empty history)
            hist = []
        
        # Lưu lại original history (có thể đã cắt)
        original_hist = hist[-self.max_history:] if len(hist) > self.max_history else hist
        hist_len = len(original_hist)
        
        # --- B. Load History Content Features ---
        # Initialize arrays for history features
        history_text_feats = np.zeros((self.max_history, Config.TEXT_DIM), dtype=np.float32)
        history_video_feats = np.zeros((self.max_history, Config.VIDEO_DIM), dtype=np.float32)
        
        # Fill in actual features for non-padding positions
        for i, pid_int in enumerate(original_hist):
            if pid_int != 0:  # Skip padding
                t_feat, v_feat = self._get_content_features(pid_int)
                history_text_feats[i] = t_feat
                history_video_feats[i] = v_feat
        
        # Padding History PIDs (for mask)
        if hist_len < self.max_history:
            hist_padded = original_hist + [0] * (self.max_history - hist_len)
        else:
            hist_padded = original_hist
            
        # --- C. Xử lý Current Item Features ---
        pid_int = row['pid']
        t_feat, v_feat = self._get_content_features(pid_int)

        return {
            # User Tensors (NO user_id in pure content)
            'gender': torch.tensor(row['gender'], dtype=torch.long),
            'fre_city': torch.tensor(row['fre_city'], dtype=torch.long),
            'fre_community_type': torch.tensor(row['fre_community_type'], dtype=torch.long),
            'fre_city_level': torch.tensor(row['fre_city_level'], dtype=torch.long),
            'age': torch.tensor(row['age'], dtype=torch.float),
            'mod_price': torch.tensor(row['mod_price'], dtype=torch.float),
            'p_hour': torch.tensor(row['p_hour'], dtype=torch.long),
            'p_dow': torch.tensor(row['p_dow'], dtype=torch.long),
            
            # History (PIDs for mask + content features)
            'history_pids': torch.tensor(hist_padded, dtype=torch.long),
            'history_text_feats': torch.tensor(history_text_feats, dtype=torch.float),
            'history_video_feats': torch.tensor(history_video_feats, dtype=torch.float),
            
            # Item Tensors (NO pid in pure content)
            'text_feat': torch.tensor(t_feat, dtype=torch.float),
            'video_feat': torch.tensor(v_feat, dtype=torch.float),
            
            # Item Metadata
            'author_fans_count': torch.tensor(row.get('author_fans_count', 0.0), dtype=torch.float),
            'duration': torch.tensor(row.get('duration', 0.0), dtype=torch.float),
        }