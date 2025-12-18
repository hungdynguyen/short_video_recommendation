import os
from dotenv import load_dotenv
import torch
import numpy as np
import json
import pandas as pd

from qdrant_client import QdrantClient
from transformers import AutoModel

from src.models.config import Config
from src.models.user_data import UserData
from src.constants import CONFIG_PATH_PIPELINE
from src.models.hyperparams import ConfigHyperparams
from src.inference_models import UserTower, ItemTower, RecModel

load_dotenv()

# Load configuration
config_pipeline = Config.from_yaml(CONFIG_PATH_PIPELINE)

# Connect to Qdrant
search_config = config_pipeline.search_config
qdrant_client = QdrantClient(
    url=f"{os.getenv('QDRANT_HOST')}:{int(os.getenv('QDRANT_PORT', 6333))}", 
    api_key=f"{os.getenv('QDRANT_API_KEY')}",
)


# Cache features to avoid reloading
_feature_cache = {}
_demographic_recommendation_cache = {}

# Load feature mappings and content
def _load_content_features():
    """Load text and video features with ID mappings."""
    text_feats = np.load(ConfigHyperparams.TEXT_FEATS_PATH).astype(np.float32)
    video_feats = np.load(ConfigHyperparams.VIDEO_FEATS_PATH).astype(np.float32)
    text_ids = np.load(ConfigHyperparams.TEXT_IDS_PATH)
    video_ids = np.load(ConfigHyperparams.VIDEO_IDS_PATH)
    
    # Create ID -> Index mappings
    text_map = {str(pid): i for i, pid in enumerate(text_ids)}
    video_map = {str(pid): i for i, pid in enumerate(video_ids)}
    
    return text_feats, video_feats, text_map, video_map

def _load_demographic_recommendations():
    """Load precomputed demographic-based recommendations."""
    if not _demographic_recommendation_cache:
        interaction_df = pd.read_csv(ConfigHyperparams.INTERACTION_DATA_DIR)

        # Create age group column
        interaction_df['age_group'] = interaction_df['age'].apply(
            lambda x: 'old' if x >= interaction_df['age'].mean() else 'young'
        )
        
        # Get top 10 videos for each gender-age combination
        recommendations = {}
        for gender in ['M', 'F']:
            recommendations[gender] = {}
            for age_group in ['young', 'old']:
                filtered_df = interaction_df[
                    (interaction_df['gender'] == gender) & 
                    (interaction_df['age_group'] == age_group)
                ]
                top_videos = filtered_df['pid'].value_counts().head(10).index.tolist()
                recommendations[gender][age_group] = top_videos

        _demographic_recommendation_cache.update(recommendations)
    
    return _demographic_recommendation_cache



def _get_feature_cache():
    """Lazy load and cache content features."""
    if not _feature_cache:
        text_feats, video_feats, text_map, video_map = _load_content_features()
        _feature_cache['text_feats'] = text_feats
        _feature_cache['video_feats'] = video_feats
        _feature_cache['text_map'] = text_map
        _feature_cache['video_map'] = video_map
    return _feature_cache


def _get_content_features(pid_int, int2pid_map, device):
    """Get text and video features for a given video ID."""
    cache = _get_feature_cache()
    text_feats = cache['text_feats']
    video_feats = cache['video_feats']
    text_map = cache['text_map']
    video_map = cache['video_map']
    
    real_pid_str = int2pid_map.get(pid_int, "Unknown")
    
    # Get text feature
    t_idx = text_map.get(real_pid_str)
    t_feat = torch.tensor(text_feats[t_idx] if t_idx is not None else np.zeros(ConfigHyperparams.TEXT_DIM, dtype=np.float32), 
                         dtype=torch.float, device=device)
    
    # Get video feature
    v_idx = video_map.get(real_pid_str)
    v_feat = torch.tensor(video_feats[v_idx] if v_idx is not None else np.zeros(ConfigHyperparams.VIDEO_DIM, dtype=np.float32),
                         dtype=torch.float, device=device)
    
    return t_feat, v_feat


def create_user_batch(
    gender: int,
    age: float,
    city: int,
    community_type: int,
    city_level: int,
    price: float,
    hour: int,
    day_of_week: int,
    history_pids: list = None,
    int2pid_map: dict = None,
    mappings: dict = None,
    device: str = 'cpu'
):
    """
    Create a user batch for inference (works for both old and new users).
    
    Args:
        gender: 0 (female) or 1 (male)
        age: Normalized age [0, 1]
        city: City ID
        community_type: Community type (0-3)
        city_level: City level (1-6)
        price: Normalized price [0, 1]
        hour: Hour of day (0-23)
        day_of_week: Day of week (0=Monday, 6=Sunday)
        history_pids: List of integer PIDs (None for cold-start)
        int2pid_map: Mapping from int PID to string PID (will be created from mappings if None)
        mappings: Mappings dict (loaded from JSON) to create int2pid_map if needed
        device: 'cpu' or 'cuda'
    
    Returns:
        Dictionary with batch tensors
    """
    if device == 'cuda':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    
    # Create int2pid_map if not provided
    if int2pid_map is None:
        if mappings is None:
            import json
            with open(ConfigHyperparams.MAPPINGS_JSON, 'r') as f:
                mappings = json.load(f)
        int2pid_map = {v: k for k, v in mappings['pid'].items()}
        int2pid_map = {k: v.replace('.mp4', '') for k, v in int2pid_map.items()}
    
    # Handle history
    if history_pids is None or (isinstance(history_pids, float) and np.isnan(history_pids)):
        history_pids = []
    elif isinstance(history_pids, np.ndarray):
        history_pids = history_pids.tolist()
    
    # Trim to max history
    max_history = ConfigHyperparams.MAX_HISTORY
    original_hist = history_pids[-max_history:] if len(history_pids) > max_history else history_pids
    hist_len = len(original_hist)
    
    # Load content features for history
    history_text_feats = np.zeros((max_history, ConfigHyperparams.TEXT_DIM), dtype=np.float32)
    history_video_feats = np.zeros((max_history, ConfigHyperparams.VIDEO_DIM), dtype=np.float32)
    
    for i, pid_int in enumerate(original_hist):
        if pid_int != 0:
            t_feat, v_feat = _get_content_features(pid_int, int2pid_map, device)
            history_text_feats[i] = t_feat.cpu().numpy()
            history_video_feats[i] = v_feat.cpu().numpy()
    
    # Pad history PIDs
    hist_padded = original_hist + [0] * (max_history - hist_len) if hist_len < max_history else original_hist
    
    batch = {
        'gender': torch.tensor([gender], dtype=torch.long, device=device),
        'fre_city': torch.tensor([city], dtype=torch.long, device=device),
        'fre_community_type': torch.tensor([community_type], dtype=torch.long, device=device),
        'fre_city_level': torch.tensor([city_level], dtype=torch.long, device=device),
        'age': torch.tensor([age], dtype=torch.float, device=device),
        'mod_price': torch.tensor([price], dtype=torch.float, device=device),
        'p_hour': torch.tensor([hour], dtype=torch.long, device=device),
        'p_dow': torch.tensor([day_of_week], dtype=torch.long, device=device),
        
        # History
        'history_pids': torch.tensor([hist_padded], dtype=torch.long, device=device),
        'history_text_feats': torch.tensor([history_text_feats], dtype=torch.float, device=device),
        'history_video_feats': torch.tensor([history_video_feats], dtype=torch.float, device=device),
    }
    
    return batch

def search_similar_videos(user_emb: torch.Tensor, client: QdrantClient, top_k: int = 10, device: str = 'cpu'):
    """
    Search Qdrant for videos similar to the user embedding.
    """
    # Ensure embedding is 1D
    if user_emb.dim() == 2:
        user_emb = user_emb.squeeze(0)
    
    # Convert to numpy and then to list for Qdrant
    user_emb_list = user_emb.cpu().detach().numpy().tolist()
    
    # Search in Qdrant
    search_results = client.query_points(
        collection_name="videos",
        query=user_emb_list,
        limit=top_k,
        with_payload=True  # Include payload (metadata)
    )
    
    # Format results
    recommendations = []
    for point in search_results.points:
        rec = {
            'pid': point.payload.get('pid'),
            'score': point.score,
            'author_fans_count': point.payload.get('author_fans_count', 0.0),
            'video_id': point.payload.get('video_id', None),
            'duration': point.payload.get('duration', 0.0)
        }
        recommendations.append(rec)
    
    return recommendations

def search_demographic_recommendations(user_gender: int, user_age: float):
    
    cache = _load_demographic_recommendations()
    print(cache)

    gender = 'M' if user_gender == 1 else 'F'

    gender_cache = cache.get(gender, {})

    age_group = 'old' if user_age >= ConfigHyperparams.AGE_THRESHOLD else 'young'

    recommendations = gender_cache.get(age_group, [])

    return recommendations


def predict(user_data: dict, model: torch.nn.Module, device: str = 'cpu', top_k: int = 10):
    """
    Generate recommendations for a user (old or new).
    
    Args:
        user_data: Dictionary with user profile
            Required: gender, age, city, community_type, city_level, price, hour, day_of_week
            Optional: history_pids (for returning users)
        model: Recommendation model with user_tower and item_tower
        device: 'cpu' or 'cuda'
        top_k: Number of top recommendations to return
    
    Returns:
        List of top-k video recommendations with scores and metadata
    """
    # Load mappings once
    import json
    with open(ConfigHyperparams.MAPPINGS_JSON, 'r') as f:
        mappings = json.load(f)
    
    int2pid_map = {v: k for k, v in mappings['pid'].items()}
    int2pid_map = {k: v.replace('.mp4', '') for k, v in int2pid_map.items()}
    
    # Create batch
    batch = create_user_batch(
        gender=user_data['gender'],
        age=user_data['age'],
        city=user_data['city'],
        community_type=user_data['community_type'],
        city_level=user_data['city_level'],
        price=user_data['price'],
        hour=user_data['hour'],
        day_of_week=user_data['day_of_week'],
        history_pids=user_data.get('history_pids', None),
        int2pid_map=int2pid_map,
        mappings=mappings,
        device=device
    )

    with torch.no_grad():
        # Debug: Print batch shapes
        print("Batch shapes:")
        for key, val in batch.items():
            print(f"  {key}: {val.shape}")
        
        user_emb = model.user_tower(batch)  # [1, embedding_dim]
    
    # Search Qdrant for similar videos
    results = search_similar_videos(user_emb, qdrant_client, top_k=top_k)

    demographic_results = search_demographic_recommendations(
        user_data['gender'], user_data['age']
    )

    print(demographic_results)

    results.extend([{'video_id': video_id, 'score': 'DEMOGRAPHIC_MATCH'} for video_id in demographic_results if video_id not in [r['video_id'] for r in results]])

    return results


def run_serving_pipeline(user_data: UserData):
    

    with open(ConfigHyperparams.MAPPINGS_JSON, 'r') as f:
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


    user_tower = UserTower(ConfigHyperparams, mapping_sizes).to(ConfigHyperparams.DEVICE)
    item_tower = ItemTower(ConfigHyperparams).to(ConfigHyperparams.DEVICE)
    rec_model = RecModel(user_tower, item_tower).to(ConfigHyperparams.DEVICE)

    checkpoint = torch.load(ConfigHyperparams.CHECKPOINT_PATH, map_location=ConfigHyperparams.DEVICE, weights_only=True)
    rec_model.load_state_dict(checkpoint)
    rec_model.eval()

    recs = predict(
        model=rec_model,
        device=ConfigHyperparams.DEVICE,
        user_data=user_data.model_dump()
    )

    return recs




