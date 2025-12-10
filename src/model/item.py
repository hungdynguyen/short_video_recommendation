"""Pure Content-Based ItemTower - No ID embeddings, only content features."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ItemTower(nn.Module):
    def __init__(self, config):
        """Pure content-based ItemTower without ID embeddings.
        
        Args:
            config: Config object with COMBINED_DIM, EMBED_DIM, DROPOUT
        """
        super().__init__()
        
        # 1. Projection for text features (1024 → 256) - bge-m3 output
        self.text_proj = nn.Sequential(
            nn.Linear(config.TEXT_DIM, 512),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(512, 256)
        )
        
        # 2. Projection for video features (768 → 192)
        self.video_proj = nn.Sequential(
            nn.Linear(config.VIDEO_DIM, 384),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(384, 192)
        )
        
        # 3. Projection for numerical metadata (author_fans_count, duration)
        self.num_proj = nn.Linear(2, 64)
        
        # 4. Final MLP: 256 (text) + 192 (video) + 64 (num) = 512 → 512
        fusion_in = 256 + 192 + 64
        
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_in, config.EMBED_DIM),  # 512 → 512 direct
            nn.ReLU(),
            nn.Dropout(config.DROPOUT)
        )
        
        # REMOVED: Custom weight initialization (PyTorch default is better)

    def forward(self, batch):
        """
        Args:
            batch: dict with keys:
                - text_feat: [B, 768]
                - video_feat: [B, 768]
                - author_fans_count: [B] (float)
                - duration: [B] (float)
        
        Returns:
            item_emb: [B, 512] normalized embeddings
        """
        # Transform content features
        # Note: Both text and video features are already L2 normalized during extraction
        text_transformed = self.text_proj(batch['text_feat'])    # [B, 256]
        video_transformed = self.video_proj(batch['video_feat'])  # [B, 192]
        
        # Numerical metadata (already MinMax scaled, no NaN/Inf)
        nums = torch.stack([batch['author_fans_count'], batch['duration']], dim=1)  # [B, 2]
        num_transformed = self.num_proj(nums)  # [B, 64]
        
        # Concatenate all features
        concat = torch.cat([
            text_transformed, 
            video_transformed, 
            num_transformed
        ], dim=1)  # [B, 512]
        
        # Final projection
        item_raw = self.fusion_mlp(concat)
        
        # L2 normalization (clamp prevents division by zero for edge cases)
        norm = torch.norm(item_raw, p=2, dim=1, keepdim=True).clamp(min=1e-8)
        return item_raw / norm