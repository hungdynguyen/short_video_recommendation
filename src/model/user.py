import torch
import torch.nn as nn
import torch.nn.functional as F

class UserTower(nn.Module):
    def __init__(self, config, mapping_sizes):
        super().__init__()
        
        # Embedding Layers (Sizes lấy từ mapping file)
        # +2 để dư chỗ cho padding(0) và unknown
        self.user_emb = nn.Embedding(mapping_sizes.get('user_id', 10000) + 2, 128)
        self.gender_emb = nn.Embedding(mapping_sizes.get('gender', 5) + 2, 16)
        self.city_emb = nn.Embedding(mapping_sizes.get('fre_city', 100) + 2, 32)
        self.comm_emb = nn.Embedding(mapping_sizes.get('fre_community_type', 10) + 2, 16)
        self.level_emb = nn.Embedding(mapping_sizes.get('fre_city_level', 10) + 2, 8)
        
        self.hour_emb = nn.Embedding(25, 16)
        self.dow_emb = nn.Embedding(8, 16)
        
        self.num_proj = nn.Linear(2, 32) # age, price
        
        # History Encoder
        num_pids = mapping_sizes.get('pid', 50000) + 2
        self.history_pid_emb = nn.Embedding(num_pids, 256)
        self.history_attn = nn.MultiheadAttention(embed_dim=256, num_heads=4, batch_first=True)
        
        # Calculate Total Input Dim
        # 128+16+32+16+8 + 16+16 + 32 + 256(history) = 520
        input_dim = 128 + 16 + 32 + 16 + 8 + 16 + 16 + 32 + 256
        
        # Deep MLP to 1024
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            nn.SiLU(),
            nn.Linear(2048, config.EMBED_DIM) # 1024
        )

    def forward(self, batch):
        # Embeddings
        u = self.user_emb(batch['user_id'])
        g = self.gender_emb(batch['gender'])
        c = self.city_emb(batch['fre_city'])
        cm = self.comm_emb(batch['fre_community_type'])
        l = self.level_emb(batch['fre_city_level'])
        h = self.hour_emb(batch['p_hour'])
        d = self.dow_emb(batch['p_dow'])
        
        # Numerical
        nums = torch.stack([batch['age'], batch['mod_price']], dim=1)
        n = self.num_proj(nums)
        
        # History Attention
        hist = self.history_pid_emb(batch['history_pids']) # [B, 50, 256]
        padding_mask = (batch['history_pids'] == 0)
        attn_out, _ = self.history_attn(hist, hist, hist, key_padding_mask=padding_mask)
        attn_out = torch.nan_to_num(attn_out, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Mean Pooling (ignoring padding); handle empty history to avoid NaN
        mask_expanded = (~padding_mask).unsqueeze(-1).float()
        valid_count = mask_expanded.sum(dim=1)  # [B, 1]
        hist_rep = (attn_out * mask_expanded).sum(dim=1)
        hist_rep = hist_rep / valid_count.clamp(min=1.0)
        # If no history (valid_count==0), set to zero vector
        hist_rep = torch.where(valid_count > 0, hist_rep, torch.zeros_like(hist_rep))
        hist_rep = torch.nan_to_num(hist_rep, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Concat & MLP
        concat = torch.cat([u, g, c, cm, l, h, d, n, hist_rep], dim=1)
        # Avoid NaN when vector norm is near zero
        return F.normalize(self.mlp(concat), p=2, dim=1, eps=1e-8)