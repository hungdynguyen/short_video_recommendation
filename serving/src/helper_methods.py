import torch 
import numpy as np
import pickle
from qdrant_client.models import Distance, VectorParams

from src.constants import CONFIG_VIDEO_METADATA_FILE
from src.models.hyperparams import ConfigHyperparams

with open(CONFIG_VIDEO_METADATA_FILE, 'rb') as f:
    video_metadata = pickle.load(f)

def get_pid_list(mappings, size):
    pid_list = []
    text_ids = np.load(ConfigHyperparams.TEXT_IDS_PATH)
    int2pid = {v: k for k, v in mappings['pid'].items()}
    int2pid = {k: v.replace('.mp4', '') for k, v in int2pid.items()}
    for i in range(size):
        real_pid = text_ids[i]
        # Find the integer key that maps to this real_pid
        for int_key, pid_str in int2pid.items():
            if pid_str == real_pid:
                pid_list.append(int_key)
                break
    
    return pid_list

def get_video_metadata(pid):
    key = list(video_metadata.keys())[pid]
    
    return {
        'author_fans_count': torch.tensor([video_metadata[key]['author_fans_count']], dtype=torch.float, device=ConfigHyperparams.DEVICE),
        'duration': torch.tensor([video_metadata[key]['duration']], dtype=torch.float, device=ConfigHyperparams.DEVICE)
    }

def create_collection(name, client, embedding_dim):
    try:
        client.get_collection(collection_name=name)
        print(f"Qdrant collection '{name}' already exists.")
    except:
        client.recreate_collection(
            collection_name=name,
            vectors_config=VectorParams(
                size=embedding_dim,
                distance=Distance.COSINE
            )
        )

