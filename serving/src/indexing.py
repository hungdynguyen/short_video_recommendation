import os
from dotenv import load_dotenv
import torch
import json
import numpy as np


from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from src.models.config import Config
from src.constants import CONFIG_PATH_PIPELINE
from src.helper_methods import get_pid_list, get_video_metadata, create_collection
from src.inference_models import UserTower, ItemTower, RecModel
from src.models.hyperparams import ConfigHyperparams

# Load environment variables from .env
load_dotenv()

# Load configuration
config_pipeline = Config.from_yaml(CONFIG_PATH_PIPELINE)
search_config = config_pipeline.search_config

qdrant_client = QdrantClient(
    url=f"{os.getenv('QDRANT_HOST')}:{int(os.getenv('QDRANT_PORT', 6333))}", 
    api_key=f"{os.getenv('QDRANT_API_KEY')}",
    timeout=300
)

def process_item_embeddings(text_feats, video_feats, rec_model, pid_list):
    num_items = text_feats.size(0)
    item_embs = []

    for i in range(0, num_items):
        batch_text = text_feats[i].unsqueeze(0)  # [1, text_dim]
        batch_video = video_feats[i].unsqueeze(0)  # [1, video_dim]
        
        B = batch_text.size(0)
        item_batch = {
            'text_feat': batch_text,
            'video_feat': batch_video,
            **get_video_metadata(pid_list[i])
        }
        
        item_emb = rec_model.item_tower(item_batch)  # [B, 512]
        item_embs.append(item_emb)

    item_embs = torch.cat(item_embs, dim=0)  # [num_items, 512]

    video_embeddings_with_metadata = []

    for idx in range(len(pid_list)):
        video_embeddings_with_metadata.append({
            'pid': pid_list[idx],
            'embedding': item_embs[idx].cpu().detach().numpy(),  # [512]
            **get_video_metadata(pid_list[idx])
        })
    return video_embeddings_with_metadata

def index_videos(video_data, pid_mapping=None):
    """Index all videos into the vector database."""

    batch_size = 128
    idx_to_video_id = {v: k for k, v in pid_mapping.items()} if pid_mapping else None

    total_videos = len(video_data)
    for batch_start in range(0, total_videos, batch_size):
        batch_end = min(batch_start + batch_size, total_videos)
        batch_items = video_data[batch_start:batch_end]

        qdrant_client.upsert(
            collection_name="videos",
            points=[
                {
                    "id": batch_start + idx,  # Add unique ID
                    "vector": item['embedding'].tolist(),
                    "payload": {
                        "pid": item['pid'],
                        "author_fans_count": float(item['author_fans_count'].squeeze(0).numpy()),
                        "duration": float(item['duration'].squeeze(0).numpy()),
                        "video_id": idx_to_video_id[item['pid']] if idx_to_video_id else None
                    }
                }
                for idx, item in enumerate(batch_items)
            ]
        )
        print(f"Indexed batch {batch_start}-{batch_end}/{total_videos}")
    print("Videos indexed successfully.")

def delete_collection(collection_name: str):
    """Delete a collection from Qdrant."""
    qdrant_client.delete_collection(collection_name=collection_name)
    print(f"Collection '{collection_name}' deleted successfully.")


if __name__ == "__main__":

    #delete_collection("videos")

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

    text_feats = torch.from_numpy(np.load(ConfigHyperparams.TEXT_FEATS_PATH).astype(np.float32)).to(ConfigHyperparams.DEVICE)
    video_feats = torch.from_numpy(np.load(ConfigHyperparams.VIDEO_FEATS_PATH).astype(np.float32)).to(ConfigHyperparams.DEVICE)

    create_collection(
        name="videos",
        client=qdrant_client,
        embedding_dim=search_config['parameters']['dimension']
    )

    pid_list = get_pid_list(mappings, text_feats.size(0))
    embeddings = process_item_embeddings(text_feats, video_feats, rec_model, pid_list)

    index_videos(embeddings, pid_mapping=mappings['pid'])