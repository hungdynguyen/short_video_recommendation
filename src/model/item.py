import torch
import torch.nn as nn
import torch.nn.functional as F


class ItemTower(nn.Module):
    def __init__(self, num_items, config):
        super().__init__()
        
        # 1. ID Embedding
        self.item_id_emb = nn.Embedding(num_items + 1, 256)
        
        # 2. Projection head for concatenated text+video features
        # Input: 256 (ID) + 1536 (text 768 + video 768) = 1792
        # Output: 1024
        fusion_in = 256 + config.COMBINED_DIM
        
        self.projection_head = nn.Sequential(
            nn.Linear(fusion_in, 1280),
            nn.ReLU(),
            nn.Linear(1280, config.EMBED_DIM)  # 1024
        )

    def forward(self, batch):
        # batch['pid']: [B]
        # batch['text_feat']: [B, 768]
        # batch['video_feat']: [B, 768]
        
        id_v = self.item_id_emb(batch['pid'])  # [B, 256]
        
        # Concatenate text and video features
        combined_feat = torch.cat([batch['text_feat'], batch['video_feat']], dim=1)  # [B, 1536]
        
        # Concatenate ID embedding with combined features
        concat = torch.cat([id_v, combined_feat], dim=1)  # [B, 1792]
        
        # Project to final embedding
        # Avoid NaN when vector is all-zero by adding eps in normalization
        return F.normalize(self.projection_head(concat), p=2, dim=1, eps=1e-8)