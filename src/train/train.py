import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Ensure project root on sys.path when running as script (python src/train/train.py)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import các modules ta vừa viết
from src.config import Config
from src.dataset import FullTrainingDataset
from src.model import UserTower, ItemTower, RecModel



def main():
    # 0. Setup Environment
    os.makedirs(Config.MODEL_SAVE_PATH, exist_ok=True)
    print(f"Running on: {Config.DEVICE}")
    print(f"Embedding Dimension: {Config.EMBED_DIM}")

    # 1. Init Dataset
    # Lưu ý: Class này tự động load mappings và numpy features bên trong
    print("Initializing Dataset...")
    dataset = FullTrainingDataset(
        parquet_path=Config.TRAIN_PARQUET,
        mappings_path=Config.MAPPINGS_JSON
    )
    
    loader = DataLoader(
        dataset, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=Config.NUM_WORKERS,
        pin_memory=True
    )
    
    # Lấy sizes từ mappings để init embedding layers chính xác
    # mappings structure: {'user_id': {'u1': 1, ...}, ...}
    mapping_sizes = {k: len(v) for k, v in dataset.mappings.items()}
    print(f"Vocab Sizes: {mapping_sizes}")

    # 2. Init Models
    print("Building Models...")
    user_tower = UserTower(config=Config, mapping_sizes=mapping_sizes)
    item_tower = ItemTower(config=Config, num_items=mapping_sizes.get('pid', 50000))
    
    model = RecModel(user_tower, item_tower).to(Config.DEVICE)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LR, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()
    
    # 3. Training Loop
    print("Start Training...")
    
    for epoch in range(Config.EPOCHS):
        model.train()
        total_loss = 0
        progress_bar = tqdm(loader, desc=f"Epoch {epoch+1}/{Config.EPOCHS}")
        
        for step, batch in enumerate(progress_bar):
            # Move batch to GPU
            batch = {k: v.to(Config.DEVICE) for k, v in batch.items()}
            
            optimizer.zero_grad()
            
            # Forward Pass
            # Logits shape: [Batch_Size, Batch_Size]
            logits = model(batch)
            
            # Labels: Đường chéo (0, 1, 2...)
            labels = torch.arange(logits.size(0)).to(Config.DEVICE)
            
            # Compute Loss
            loss = criterion(logits, labels)
            
            # Backward
            loss.backward()
            optimizer.step()
            
            # Logging
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        avg_loss = total_loss / len(loader)
        print(f"--> Epoch {epoch+1} Finished. Avg Loss: {avg_loss:.4f}")
        
        # Save Checkpoint
        save_path = os.path.join(Config.MODEL_SAVE_PATH, f"two_tower_ep{epoch+1}.pth")
        torch.save(model.state_dict(), save_path)
        print(f"Saved model to {save_path}")

if __name__ == "__main__":
    main()