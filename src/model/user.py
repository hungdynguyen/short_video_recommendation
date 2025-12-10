"""Pure Content-Based UserTower - No user/item ID embeddings, only content features for history."""
import torch
import torch.nn as nn
import torch.nn.functional as F

class UserTower(nn.Module):
    def __init__(self, config, mapping_sizes):
        """Pure content-based UserTower without user_id or history_pid embeddings.
        Uses one-hot encoding + MLP instead of random embeddings.
        
        Args:
            config: Config object
            mapping_sizes: Dict with sizes for categorical features
        """
        super().__init__()
        
        # Store mapping sizes for one-hot encoding (must match actual vocab from data)
        # For one-hot encoding, we only need the actual vocab size (no +2 for padding/unknown)
        # The data preprocessing already handles mapping to 0-indexed values
        self.gender_size = mapping_sizes.get('gender', 2)
        self.city_size = mapping_sizes.get('fre_city', 293)
        self.comm_size = mapping_sizes.get('fre_community_type', 4)
        self.level_size = mapping_sizes.get('fre_city_level', 7)
        self.hour_size = 24  # Hours: 0-23
        self.dow_size = 7    # Days of week: 0-6
        
        # One-hot → Projection (deterministic transformation)
        self.gender_proj = nn.Linear(self.gender_size, 16)
        self.city_proj = nn.Linear(self.city_size, 32)
        self.comm_proj = nn.Linear(self.comm_size, 16)
        self.level_proj = nn.Linear(self.level_size, 8)
        self.hour_proj = nn.Linear(self.hour_size, 16)
        self.dow_proj = nn.Linear(self.dow_size, 16)
        
        # Numerical features (age, price)
        self.num_proj = nn.Linear(2, 32)
        
        # History: Transform content features instead of learned embeddings
        # Text: 1024 → 128, Video: 768 → 128
        self.history_text_proj = nn.Linear(1024, 128)
        self.history_video_proj = nn.Linear(config.VIDEO_DIM, 128)
        
        # TransformerEncoder for history (with safeguards against NaN)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256,  # 128 text + 128 video
            nhead=4,  # Keep small for limited data
            dim_feedforward=512,  # Conservative FFN size
            dropout=config.DROPOUT,
            activation='relu',
            batch_first=True,
            norm_first=True  # Pre-norm for stability
        )
        self.history_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2,  # 2 layers sufficient for 21K samples
            norm=nn.LayerNorm(256)
        )
        
        # Calculate Total Input Dim
        # 16+32+16+8 + 16+16 + 32 + 256(history: 128 text + 128 video) = 392
        input_dim = 16 + 32 + 16 + 8 + 16 + 16 + 32 + 256
        
        # Simplified MLP to EMBED_DIM (512) - NO BatchNorm for stability
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 768),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(768, config.EMBED_DIM)  # 512
        )
        
        # REMOVED: Custom weight initialization (was causing NaN after first backward)
        # PyTorch's default initialization is more stable

    def forward(self, batch):
        """
        Args:
            batch: dict with keys:
                - gender, fre_city, fre_community_type, fre_city_level: [B]
                - p_hour, p_dow: [B]
                - age, mod_price: [B] (float)
                - history_text_feats: [B, max_history, 1024]
                - history_video_feats: [B, max_history, 768]
                - history_pids: [B, max_history] (for padding mask only)
        
        Returns:
            user_emb: [B, 512] normalized embeddings
        """
        # One-hot encode categorical features → Project to dense
        # Note: Preprocessing ensures values are already in valid range [0, vocab_size-1]
        g_onehot = F.one_hot(batch['gender'].long(), num_classes=self.gender_size).float()
        c_onehot = F.one_hot(batch['fre_city'].long(), num_classes=self.city_size).float()
        cm_onehot = F.one_hot(batch['fre_community_type'].long(), num_classes=self.comm_size).float()
        l_onehot = F.one_hot(batch['fre_city_level'].long(), num_classes=self.level_size).float()
        h_onehot = F.one_hot(batch['p_hour'].long(), num_classes=self.hour_size).float()
        d_onehot = F.one_hot(batch['p_dow'].long(), num_classes=self.dow_size).float()
        
        g = self.gender_proj(g_onehot)      # [B, 16]
        c = self.city_proj(c_onehot)        # [B, 32]
        cm = self.comm_proj(cm_onehot)      # [B, 16]
        l = self.level_proj(l_onehot)       # [B, 8]
        h = self.hour_proj(h_onehot)        # [B, 16]
        d = self.dow_proj(d_onehot)         # [B, 16]
        
        # Numerical features (already MinMax scaled [0, 1], no NaN/Inf)
        nums = torch.stack([batch['age'], batch['mod_price']], dim=1)
        n = self.num_proj(nums)
        
        # History: Transform content features
        # Note: Both text and video features are already L2 normalized during extraction
        hist_text = batch['history_text_feats']
        hist_video = batch['history_video_feats']
        
        # Mask out padding positions
        padding_mask = (batch['history_pids'] == 0)  # [B, max_history]
        video_mask = (~padding_mask).unsqueeze(-1).float()  # [B, max_history, 1]
        
        # Apply mask to zero out padding (features already normalized)
        hist_video_masked = hist_video * video_mask
        
        hist_text_proj = self.history_text_proj(hist_text)   # [B, max_history, 128]
        hist_video_proj = self.history_video_proj(hist_video_masked) # [B, max_history, 128]
        hist = torch.cat([hist_text_proj, hist_video_proj], dim=-1)  # [B, max_history, 256]
        
        # Handle empty history (cold-start users)
        has_history = (~padding_mask).any(dim=1)  # [B], True if user has any history
        
        # Only run TransformerEncoder for users with history (avoid NaN with all-padding)
        attn_out = torch.zeros(hist.size(0), hist.size(1), 256, device=hist.device)
        if has_history.any():
            # Run encoder only for users with history
            hist_with_history = hist[has_history]
            mask_with_history = padding_mask[has_history]
            attn_with_history = self.history_encoder(
                hist_with_history,
                src_key_padding_mask=mask_with_history
            )
            attn_out[has_history] = attn_with_history
        
        # Mean pooling over valid positions        # Mean pooling over valid positions
        mask_expanded = video_mask  # [B, max_history, 1]
        valid_count = mask_expanded.sum(dim=1).clamp(min=1.0)  # Prevent division by zero
        hist_rep = (attn_out * mask_expanded).sum(dim=1) / valid_count  # [B, 256]
        
        # Concat & MLP (NO user_emb)
        concat = torch.cat([g, c, cm, l, h, d, n, hist_rep], dim=1)
        
        # MLP forward
        user_raw = self.mlp(concat)
        
        # L2 normalization (clamp prevents division by zero for edge cases)
        norm = torch.norm(user_raw, p=2, dim=1, keepdim=True).clamp(min=1e-8)
        return user_raw / norm