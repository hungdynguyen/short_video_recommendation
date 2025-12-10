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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")
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
        num_workers=0,  # Set to 0 for Windows compatibility
        pin_memory=True
    )
    
    # Lấy sizes từ mappings để init embedding layers chính xác
    # mappings structure: {'user_id': {'u1': 1, ...}, ...}
    mapping_sizes = {k: len(v) for k, v in dataset.mappings.items()}
    print(f"Vocab Sizes: {mapping_sizes}")

    # 2. Init Models (Pure Content-Based)
    print("Building Pure Content-Based Models...")
    user_tower = UserTower(config=Config, mapping_sizes=mapping_sizes)
    item_tower = ItemTower(config=Config)  # NO num_items parameter
    
    model = RecModel(user_tower, item_tower).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=Config.LEARNING_RATE, 
        weight_decay=Config.WEIGHT_DECAY
    )
    
    # Cosine annealing with warmup for better convergence
    total_steps = len(loader) * Config.EPOCHS
    
    def warmup_cosine_lambda(step):
        # Warmup phase
        if step < Config.WARMUP_STEPS:
            return step / Config.WARMUP_STEPS
        # Cosine annealing phase
        progress = (step - Config.WARMUP_STEPS) / (total_steps - Config.WARMUP_STEPS)
        return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_cosine_lambda)
    criterion = nn.CrossEntropyLoss()
    
    # 3. Training Loop
    print("Start Training...")
    
    for epoch in range(Config.EPOCHS):
        model.train()
        total_loss = 0
        progress_bar = tqdm(loader, desc=f"Epoch {epoch+1}/{Config.EPOCHS}")
        
        for step, batch in enumerate(progress_bar):
            # Move batch to GPU
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Debug: Check for NaN in input batch (first iteration only)
            if step == 0 and epoch == 0:
                print("\n=== Debug: Checking input batch ===")
                for k, v in batch.items():
                    if torch.is_tensor(v):
                        has_nan = torch.isnan(v).any().item()
                        has_inf = torch.isinf(v).any().item()
                        print(f"{k}: shape={v.shape}, has_nan={has_nan}, has_inf={has_inf}, min={v.min():.4f}, max={v.max():.4f}")
            
            optimizer.zero_grad()
            
            # Forward Pass
            # Logits shape: [Batch_Size, Batch_Size]
            logits = model(batch)
            
            # Debug: Check logits
            if step == 0 and epoch == 0:
                print(f"\nLogits: shape={logits.shape}, has_nan={torch.isnan(logits).any().item()}")
                print(f"Logits min={logits.min():.4f}, max={logits.max():.4f}")
            
            # Labels: Đường chéo (0, 1, 2...)
            labels = torch.arange(logits.size(0)).to(device)
            
            # Compute Loss
            loss = criterion(logits, labels)
            
            # Debug: Check loss
            if step == 0 and epoch == 0:
                print(f"Loss: {loss.item()}")
                print("=== End Debug ===\n")
            
            # Check for NaN loss and skip if found
            if torch.isnan(loss):
                print(f"WARNING: NaN loss detected at step {step}, skipping batch")
                continue
            
            # Backward
            loss.backward()
            
            # Debug gradients on first step
            if step == 0 and epoch == 0:
                print("\n=== Gradient Check ===")
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        grad_norm = param.grad.norm().item()
                        has_nan = torch.isnan(param.grad).any().item()
                        has_inf = torch.isinf(param.grad).any().item()
                        if has_nan or has_inf or grad_norm > 10:
                            print(f"{name}: grad_norm={grad_norm:.4f}, nan={has_nan}, inf={has_inf}")
                print("=== End Gradient Check ===\n")
            
            # Check for NaN/Inf gradients before optimizer step
            has_bad_grad = False
            for param in model.parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        has_bad_grad = True
                        break
            
            if has_bad_grad:
                print(f"WARNING: NaN/Inf gradient detected at step {step}, skipping batch")
                optimizer.zero_grad()
                continue
            
            # Gradient clipping (relaxed to allow better learning)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            
            optimizer.step()
            scheduler.step()  # Update learning rate with warmup
            
            # Logging
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        avg_loss = total_loss / len(loader)
        print(f"--> Epoch {epoch+1} Finished. Avg Loss: {avg_loss:.4f}")
        
        # Save Checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == Config.EPOCHS - 1:
            save_path = os.path.join(Config.MODEL_SAVE_PATH, f"two_tower_pure_content_ep{epoch+1}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Saved model to {save_path}")

if __name__ == "__main__":
    main()