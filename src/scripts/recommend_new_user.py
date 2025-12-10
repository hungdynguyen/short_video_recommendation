"""Recommend videos for a completely new user (cold-start scenario)."""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch
import numpy as np
from src.config import Config
from src.model.recommedation import RecModel
from src.model.user import UserTower
from src.model.item import ItemTower

def recommend_for_new_user(
    gender: int,  # 0 or 1
    age: float,  # normalized [0, 1]
    city: int,  # city ID from mappings
    community_type: int,  # 0-3
    city_level: int,  # 1-6
    price: float,  # normalized [0, 1]
    hour: int,  # 0-23
    day_of_week: int,  # 0-6 (Monday=0)
    top_k: int = 10,
    checkpoint_path: str = "data/models/two_tower_pure_content_ep20.pth"
):
    """
    Recommend top-K videos for a completely new user.
    
    Args:
        gender: 0 (female) or 1 (male)
        age: Normalized age [0, 1]
        city: City ID (0-292)
        community_type: Community type (0-3)
        city_level: City level (1-6)
        price: Normalized price [0, 1]
        hour: Hour of day (0-23)
        day_of_week: Day of week (0=Monday, 6=Sunday)
        top_k: Number of recommendations to return
        checkpoint_path: Path to trained model
    
    Returns:
        List of (pid, score) tuples
    """
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # 1. Load mappings to get vocab sizes
    import json
    with open(Config.MAPPINGS_JSON, 'r') as f:
        mappings = json.load(f)
    
    mapping_sizes = {
        'user_id': len(mappings['user_id']),
        'pid': len(mappings['pid']),
        'author_id': len(mappings['author_id']),
        'gender': len(mappings['gender']),
        'fre_city': len(mappings['fre_city']),
        'fre_community_type': len(mappings['fre_community_type']),
        'fre_city_level': len(mappings['fre_city_level'])
    }
    
    # 2. Load item features
    print("Loading item features...")
    text_feats = torch.from_numpy(np.load(Config.TEXT_FEATS_PATH).astype(np.float32)).to(device)
    video_feats = torch.from_numpy(np.load(Config.VIDEO_FEATS_PATH).astype(np.float32)).to(device)
    text_ids = np.load(Config.TEXT_IDS_PATH)
    
    # Create int2pid mapping
    int2pid = {v: k for k, v in mappings['pid'].items()}
    int2pid = {k: v.replace('.mp4', '') for k, v in int2pid.items()}
    
    # Create ordered list of pids matching feature indices
    pid_list = []
    for i in range(len(text_feats)):
        real_pid = text_ids[i]
        # Find the integer key that maps to this real_pid
        for int_key, pid_str in int2pid.items():
            if pid_str == real_pid:
                pid_list.append(int_key)
                break
    
    print(f"Loaded {len(text_feats)} items")
    
    # 3. Load model
    print("Loading model...")
    user_tower = UserTower(Config, mapping_sizes).to(device)
    item_tower = ItemTower(Config).to(device)
    model = RecModel(user_tower, item_tower).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint)
    model.eval()
    
    # 4. Create user batch (new user with NO history)
    batch = {
        'gender': torch.tensor([gender], dtype=torch.long, device=device),
        'fre_city': torch.tensor([city], dtype=torch.long, device=device),
        'fre_community_type': torch.tensor([community_type], dtype=torch.long, device=device),
        'fre_city_level': torch.tensor([city_level], dtype=torch.long, device=device),
        'age': torch.tensor([age], dtype=torch.float, device=device),
        'mod_price': torch.tensor([price], dtype=torch.float, device=device),
        'p_hour': torch.tensor([hour], dtype=torch.long, device=device),
        'p_dow': torch.tensor([day_of_week], dtype=torch.long, device=device),
        
        # Empty history (cold-start)
        'history_pids': torch.zeros((1, Config.MAX_HISTORY), dtype=torch.long, device=device),
        'history_text_feats': torch.zeros((1, Config.MAX_HISTORY, Config.TEXT_DIM), dtype=torch.float, device=device),
        'history_video_feats': torch.zeros((1, Config.MAX_HISTORY, Config.VIDEO_DIM), dtype=torch.float, device=device),
    }
    
    # 5. Encode user
    with torch.no_grad():
        user_emb = model.user_tower(batch)  # [1, 512]
    
    # 6. Encode all items (batch processing for efficiency)
    print("Encoding all items...")
    batch_size = 128
    num_items = len(text_feats)
    item_embs = []
    
    # Create dummy metadata (average values)
    avg_fans = torch.tensor([0.01], dtype=torch.float, device=device)  # Average normalized fans
    avg_duration = torch.tensor([0.1], dtype=torch.float, device=device)  # Average normalized duration
    
    for i in range(0, num_items, batch_size):
        end_idx = min(i + batch_size, num_items)
        batch_text = text_feats[i:end_idx]  # [B, 1024]
        batch_video = video_feats[i:end_idx]  # [B, 768]
        
        B = batch_text.size(0)
        item_batch = {
            'text_feat': batch_text,
            'video_feat': batch_video,
            'author_fans_count': avg_fans.expand(B),
            'duration': avg_duration.expand(B)
        }
        
        item_emb = model.item_tower(item_batch)  # [B, 512]
        item_embs.append(item_emb)
    
    item_embs = torch.cat(item_embs, dim=0)  # [num_items, 512]
    
    # 7. Compute similarity scores
    scores = torch.matmul(user_emb, item_embs.T)  # [1, num_items]
    scores = scores * model.logit_scale.exp()  # Apply temperature scaling
    scores = scores.squeeze(0)  # [num_items]
    
    # 8. Get top-K recommendations
    top_scores, top_indices = torch.topk(scores, k=min(top_k, len(scores)))
    
    # 9. Convert to PIDs
    recommendations = []
    for idx, score in zip(top_indices.cpu().numpy(), top_scores.cpu().detach().numpy()):
        pid = pid_list[idx]
        recommendations.append((pid, float(score)))
    
    return recommendations


if __name__ == "__main__":
    # Example: Recommend for a new user
    print("=== Recommending for New User ===\n")
    
    # Example user profile
    user_profile = {
        'gender': 1,  # Male
        'age': 0.5423728813559322,  # Middle-aged (normalized)
        'city': 225,  # City ID
        'community_type': 2,  # Community type
        'city_level': 2,  # Tier-3 city
        'price': 0.045977011494252866,  # Price preference
        'hour': 13,  # 8 PM
        'day_of_week': 4,  # Friday
    }
    
    print("User Profile:")
    for k, v in user_profile.items():
        print(f"  {k}: {v}")
    
    print("\nGenerating recommendations...\n")
    
    # Get recommendations
    recommendations = recommend_for_new_user(**user_profile, top_k=10)
    
    # Print results
    print("Top 10 Recommended Videos:")
    print("-" * 50)
    for i, (pid, score) in enumerate(recommendations, 1):
        print(f"{i}. PID: {pid:4d} | Score: {score:.4f}")
