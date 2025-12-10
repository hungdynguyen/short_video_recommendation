"""Evaluate two-tower model on validation set with loss and Recall@K.

Usage (from repo root, venv active):
    python -m src.scripts.eval_recall
"""
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Config
from src.dataset import FullTrainingDataset
from src.model import UserTower, ItemTower, RecModel


def latest_checkpoint(model_dir: Path) -> Path:
    """Return newest .pth in model_dir."""
    candidates = sorted(model_dir.glob("*.pth"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No .pth checkpoints found in {model_dir}")
    return candidates[0]


def recall_at_k(logits: torch.Tensor, k: int) -> float:
    """Compute Recall@K using in-batch negatives (diagonal is positive)."""
    # logits: [B, B]; labels are 0..B-1 (diagonal)
    topk = logits.topk(k, dim=1).indices  # [B, k]
    targets = torch.arange(logits.size(0), device=logits.device).unsqueeze(1)  # [B,1]
    hits = (topk == targets).any(dim=1).float()  # [B]
    return hits.mean().item()


def evaluate_subset(parquet_path: str, ckpt_path: Path, ds_template: FullTrainingDataset) -> dict:
    """Evaluate model on a specific parquet subset."""
    ds = FullTrainingDataset(
        parquet_path=parquet_path,
        mappings_path=Config.MAPPINGS_JSON,
    )
    
    if len(ds) == 0:
        return {
            'loss': 0.0,
            'recall1': 0.0,
            'recall5': 0.0,
            'recall10': 0.0,
            'num_samples': 0
        }
    
    loader = DataLoader(
        ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    mapping_sizes = {k: len(v) for k, v in ds.mappings.items()}
    user_tower = UserTower(config=Config, mapping_sizes=mapping_sizes)
    item_tower = ItemTower(config=Config)
    model = RecModel(user_tower, item_tower).to(Config.DEVICE)

    state = torch.load(ckpt_path, map_location=Config.DEVICE)
    model.load_state_dict(state)
    model.eval()

    criterion = torch.nn.CrossEntropyLoss()

    total_loss = 0.0
    total_batches = 0
    recall1_list = []
    recall5_list = []
    recall10_list = []

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(Config.DEVICE) for k, v in batch.items()}
            logits = model(batch)  # [B,B]
            labels = torch.arange(logits.size(0), device=logits.device)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            total_batches += 1

            # Metrics
            recall1_list.append(recall_at_k(logits, 1))
            recall5_list.append(recall_at_k(logits, 5))
            recall10_list.append(recall_at_k(logits, 10))

    avg_loss = total_loss / max(1, total_batches)
    r1 = sum(recall1_list) / max(1, len(recall1_list))
    r5 = sum(recall5_list) / max(1, len(recall5_list))
    r10 = sum(recall10_list) / max(1, len(recall10_list))

    return {
        'loss': avg_loss,
        'recall1': r1,
        'recall5': r5,
        'recall10': r10,
        'num_samples': len(ds)
    }


def main() -> None:
    ckpt_path = latest_checkpoint(Path(Config.MODEL_SAVE_PATH))

    print("=== Eval Two-Tower ===")
    print(f"Device: {Config.DEVICE}")
    print(f"Checkpoint: {ckpt_path}")

    # Load template dataset for mappings
    ds_template = FullTrainingDataset(
        parquet_path=Config.VAL_PARQUET,
        mappings_path=Config.MAPPINGS_JSON,
    )

    # Evaluate on all validation data
    print("\n=== Overall Validation Set ===")
    results_all = evaluate_subset(Config.VAL_PARQUET, ckpt_path, ds_template)
    print(f"Samples: {results_all['num_samples']}")
    print(f"Loss: {results_all['loss']:.4f}")
    print(f"Recall@1:  {results_all['recall1']:.4f}")
    print(f"Recall@5:  {results_all['recall5']:.4f}")
    print(f"Recall@10: {results_all['recall10']:.4f}")

    # Evaluate on true cold-start users (NO history)
    print("\n=== True Cold-Start Users (NO History) ===")
    results_cold = evaluate_subset(Config.VAL_COLD_PARQUET, ckpt_path, ds_template)
    print(f"Samples: {results_cold['num_samples']}")
    if results_cold['num_samples'] > 0:
        print(f"Loss: {results_cold['loss']:.4f}")
        print(f"Recall@1:  {results_cold['recall1']:.4f}")
        print(f"Recall@5:  {results_cold['recall5']:.4f}")
        print(f"Recall@10: {results_cold['recall10']:.4f}")
    else:
        print("No cold-start samples found.")

    # Evaluate on warm users (with normal history)
    print("\n=== Warm Users (With History) ===")
    results_warm = evaluate_subset(Config.VAL_WARM_PARQUET, ckpt_path, ds_template)
    print(f"Samples: {results_warm['num_samples']}")
    if results_warm['num_samples'] > 0:
        print(f"Loss: {results_warm['loss']:.4f}")
        print(f"Recall@1:  {results_warm['recall1']:.4f}")
        print(f"Recall@5:  {results_warm['recall5']:.4f}")
        print(f"Recall@10: {results_warm['recall10']:.4f}")
    else:
        print("No warm samples found.")



if __name__ == "__main__":
    main()
