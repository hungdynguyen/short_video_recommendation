"""Extract video embeddings using ViT-B/16 backbone with mean pooling.

Usage (from repo root):
    python -m src.scripts.extract_video_features
"""
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from tqdm.auto import tqdm

from src.config import Config


def _load_frames(video_path: Path) -> torch.Tensor:
    """Load and uniformly sample frames from video."""
    cap = cv2.VideoCapture(str(video_path))
    num_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if num_total < Config.VIDEO_NUM_FRAMES:
        # Return zero tensor if video is too short
        sampled_frms = [torch.zeros(3, 224, 224) for _ in range(Config.VIDEO_NUM_FRAMES)]
    else:
        frame_indices = np.linspace(0, num_total - 1, Config.VIDEO_NUM_FRAMES, dtype=int)
        sampled_frms = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            rval, frame = cap.read()
            if rval:
                img = Image.fromarray(frame, mode='RGB')
                img = img.resize((224, 224))
                img = transforms.ToTensor()(img)
                img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
                sampled_frms.append(img)
            else:
                sampled_frms.append(torch.zeros(3, 224, 224))

    cap.release()
    return torch.stack(sampled_frms, dim=0)  # [num_frames, 3, 224, 224]


class GridFeatBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        self.net.heads = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # [num_frames, 768]


def main() -> None:
    input_dir = Path(Config.VIDEO_INPUT_DIR)
    assert input_dir.exists(), f"Input dir not found: {input_dir}"

    # Sort files numerically by ID
    files = sorted([f for f in input_dir.iterdir() if f.suffix.lower() == '.mp4'],
                   key=lambda x: int(x.stem))
    print(f"Found {len(files)} videos in {input_dir}")

    if not files:
        print("No videos to process. Exiting.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = GridFeatBackbone().to(device)
    model.eval()

    all_features = []
    ids = []

    for path in tqdm(files, desc="Extracting video features"):
        video_id = path.stem
        try:
            frms = _load_frames(path).to(device)
            with torch.no_grad():
                out = model(frms)  # [num_frames, 768]
                # Mean pooling across frames
                video_feature = torch.mean(out, dim=0, keepdim=True)  # [1, 768]
                
                # L2 normalize to match text embeddings (bge-m3 is auto-normalized)
                norm = torch.norm(video_feature, p=2, dim=1, keepdim=True).clamp(min=1e-8)
                video_feature = video_feature / norm

            all_features.append(video_feature.cpu().numpy())
            ids.append(video_id)
        except Exception as exc:  # noqa: BLE001
            tqdm.write(f"Error processing {video_id}: {exc}")
            # Append zero vector on error
            all_features.append(np.zeros((1, 768), dtype=np.float32))
            ids.append(video_id)

    all_features = np.concatenate(all_features, axis=0).astype(np.float32)
    os.makedirs(Path(Config.VIDEO_FEATS_PATH).parent, exist_ok=True)
    np.save(Config.VIDEO_FEATS_PATH, all_features)
    np.save(Config.VIDEO_IDS_PATH, np.array(ids))

    print(f"\nAll features saved to '{Config.VIDEO_FEATS_PATH}', shape: {all_features.shape}")
    print(f"IDs saved to {Config.VIDEO_IDS_PATH}, count={len(ids)}")


if __name__ == "__main__":
    main()
