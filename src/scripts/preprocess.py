import json
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tqdm import tqdm

from src.config import Config


def preprocess_user_data():
    """Create user-side parquet + mappings for the two-tower training step.
    
    Input: CSV file đã được preprocess bởi Preprocessing-Interactions.ipynb
           (đã có watch_ratio, engagement_score, các features đã scaled/encoded)
    Output: Train/Val parquet files với history_pids
    """
    os.makedirs(Config.PROCESSED_DIR, exist_ok=True)

    print(">>> Loading Preprocessed Data...")
    df = pd.read_csv(Config.RAW_INTERACTION_PATH)
    
    print(f">>> Loaded {len(df)} interactions")
    print(f">>> Users: {df['user_id'].nunique()}, Videos: {df['pid'].nunique()}")
    
    # ==============================================
    # INPUT ĐÃ CÓ SẴN TỪ NOTEBOOK:
    # - watch_ratio, engagement_score (đã tính)
    # - age, mod_price, author_fans_count, watch_time, duration (đã MinMax scaled)
    # - gender, fre_city, fre_community_type, fre_city_level (đã encoded bằng cat.codes)
    # ==============================================
    
    # 1. Encode User ID và PID (để lưu mapping cho inference)
    print(">>> Encoding User ID and Video ID...")
    mappings = {}
    
    # Encode user_id, pid, author_id
    for col in ['user_id', 'pid', 'author_id']:
        if col not in df.columns:
            continue
        df[col] = df[col].astype(str)
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        mappings[col] = {str(k): int(v) for k, v in zip(le.classes_, le.transform(le.classes_))}
        # +1 để dành 0 cho padding
        df[col] = df[col] + 1
    
    # Thêm categorical mappings và ENCODE về 0-indexed
    # Categorical features có thể có gaps (e.g., city: [0, 5, 10, 225])
    # Cần map về continuous [0, 1, 2, 3, ...] để one-hot encoding
    cat_cols = ['gender', 'fre_city', 'fre_community_type', 'fre_city_level']
    for col in cat_cols:
        if col in df.columns:
            # LabelEncoder to map arbitrary values → 0-indexed
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            # Save mapping: original_value → encoded_index
            mappings[col] = {str(orig): int(enc) for orig, enc in zip(le.classes_, range(len(le.classes_)))}
            print(f"  - {col}: {len(le.classes_)} unique values → 0-{len(le.classes_)-1}")
    
    # Lưu mappings
    with open(Config.MAPPINGS_JSON, "w") as f:
        json.dump(mappings, f)
    print(f">>> Saved mappings to {Config.MAPPINGS_JSON}")
    
    # 2. Xử lý thời gian (Context) - nếu chưa có trong CSV
    print(">>> Processing Time Features...")
    if 'p_dow' not in df.columns:
        df['dt'] = pd.to_datetime(df['exposed_time'], unit='s')
        df['p_dow'] = df['dt'].dt.dayofweek
    # p_hour đã có sẵn trong data
    
    # 3. Tạo Target Label
    print(">>> Creating Target Labels...")
    # Positive: watch_ratio >= 0.3 HOẶC có engagement NHƯNG KHÔNG hate
    df['target'] = (
        (
            (df['watch_ratio'] >= 0.3) | 
            (df.get('cvm_like', 0) == 1) | 
            (df.get('collect', 0) == 1) |
            (df.get('forward', 0) == 1) |
            (df.get('comment', 0) == 1) |
            (df.get('share', 0) == 1) |
            (df.get('click', 0) == 1)
        ) & (df.get('hate', 0) == 0)  
    ).astype(int)

    # 4. Tạo User History Sequence
    print(">>> Generating User History Sequence...")
    # Sắp xếp theo user và thời gian
    df = df.sort_values(['user_id', 'exposed_time'])
    
    # Group interactions by user
    # Chỉ lấy những video có tương tác TỐT để làm lịch sử
    positive_df = df[df['target'] == 1]
    user_history_map = positive_df.groupby('user_id')['pid'].apply(list).to_dict()
    
    # Tạo cột history cho mỗi dòng (là list pid TRƯỚC thời điểm đó)
    # *Lưu ý: Để đơn giản hóa cho demo, ta sẽ lấy random history hoặc 
    # dùng cách map nhanh. Trong production cần strictly time-based.*
    
    # Cách nhanh cho dataset vừa phải:
    history_col = []
    # Cache history hiện tại của user đang duyệt
    curr_user_hist = {} # {uid: [pid1, pid2]}
    
    for uid, pid, target in zip(df['user_id'], df['pid'], df['target']):
        hist = curr_user_hist.get(uid, [])
        # Lấy 50 item gần nhất
        history_col.append(hist[-50:])
        
        # Nếu tương tác tốt, thêm vào history cho dòng tiếp theo dùng
        if target == 1:
            hist.append(pid)
            curr_user_hist[uid] = hist

    df['history_pids'] = history_col
    
    # 5. Chia Train/Val với TRUE Cold-Start Split
    print(">>> Splitting Train/Val with True Cold-Start...")
    # Chỉ giữ lại cột cần thiết cho User Tower và Training
    final_cols = [
        # User features
        'user_id', 'gender', 'age', 'mod_price', 
        'fre_city', 'fre_community_type', 'fre_city_level',
        
        # Context features
        'p_dow', 'p_hour',
        
        # User history
        'history_pids',
        
        # Item features (current interaction)
        'pid', 'author_id', 
        'full_categories', 'tag_name',  
        'author_fans_count', 'duration',  
        
        # Label
        'target'
    ]
    
    # Lọc chỉ các cột tồn tại trong df
    final_cols = [col for col in final_cols if col in df.columns]
    
    # LỌC CHỈ GIỮ HIGH-QUALITY INTERACTIONS (target=1)
    # Lý do: In-batch negative sampling sẽ maximize similarity cho DIAGONAL
    # Nếu giữ target=0, model học maximize similarity cho bad interactions!
    print(f">>> Before filtering: {len(df)} interactions")
    df = df[df['target'] == 1]
    print(f">>> After filtering (target=1 only): {len(df)} high-quality interactions")
    
    # ==================================================================
    # BƯỚC 1: TÁCH COLD USERS TRƯỚC (users có ít interactions)
    # ==================================================================
    user_interaction_counts = df.groupby('user_id').size()
    sorted_users = user_interaction_counts.sort_values()  # Tăng dần
    
    # Chọn users có ≤ 5 interactions làm cold-start
    # Lý do: Users có quá ít data không đủ để model học được pattern
    cold_threshold = 5
    cold_user_mask = user_interaction_counts <= cold_threshold
    cold_users = user_interaction_counts[cold_user_mask].index.tolist()
    
    print(f"\n>>> Cold-Start User Selection (interactions ≤ {cold_threshold}):")
    print(f"    - Total users: {len(user_interaction_counts)}")
    print(f"    - Cold users: {len(cold_users)} ({len(cold_users)/len(user_interaction_counts)*100:.1f}%)")
    print(f"    - Warm users: {len(user_interaction_counts) - len(cold_users)}")
    
    # Tách data theo user groups
    df_cold_users = df[df['user_id'].isin(cold_users)].copy()
    df_warm_users = df[~df['user_id'].isin(cold_users)].copy()
    
    print(f"\n>>> Interaction Distribution:")
    print(f"    - Cold users: {len(df_cold_users)} interactions")
    print(f"    - Warm users: {len(df_warm_users)} interactions")
    
    # ==================================================================
    # BƯỚC 2: CHIA TEMPORAL CHO WARM USERS (90/10)
    # ==================================================================
    df_warm_users = df_warm_users.sort_values('exposed_time')
    split_idx = int(len(df_warm_users) * 0.9)
    train_df = df_warm_users.iloc[:split_idx]
    val_warm_df = df_warm_users.iloc[split_idx:]
    
    print(f"\n>>> Warm Users Temporal Split:")
    print(f"    - Train: {len(train_df)} samples (90% earliest)")
    print(f"    - Val:   {len(val_warm_df)} samples (10% latest)")
    
    # ==================================================================
    # BƯỚC 3: COLD USERS → TOÀN BỘ VÀO VAL (TRUE COLD-START)
    # ==================================================================
    # Cold users KHÔNG BAO GIỜ xuất hiện trong train set
    # Model sẽ phải generalize dựa vào demographics + content features
    val_cold_df = df_cold_users.copy()
    
    # Xóa history để giả lập user mới hoàn toàn
    val_cold_df['history_pids'] = val_cold_df['history_pids'].apply(lambda x: [])
    
    print(f"\n>>> Cold Users (True Cold-Start):")
    print(f"    - Val:   {len(val_cold_df)} samples (100% of cold users)")
    print(f"    - Train: 0 samples (NEVER seen in training!)")
    print(f"    - History: ZERO (simulating brand new users)")
    
    # Tạo val_df tổng (backward compatibility)
    val_df = pd.concat([val_warm_df, val_cold_df], ignore_index=True)
    
    print(f"\n>>> Final Dataset Summary:")
    print(f"    - Train:     {len(train_df):,} samples (warm users only)")
    print(f"    - Val Total: {len(val_df):,} samples")
    print(f"      ├─ Warm:   {len(val_warm_df):,} samples (with history)")
    print(f"      └─ Cold:   {len(val_cold_df):,} samples (NO history, unseen users)")
    
    # Lưu các parquet files
    train_df.to_parquet(Config.TRAIN_PARQUET)
    val_df.to_parquet(Config.VAL_PARQUET)  # Vẫn giữ file tổng cho backward compatibility
    val_cold_df.to_parquet(Config.VAL_COLD_PARQUET)
    val_warm_df.to_parquet(Config.VAL_WARM_PARQUET)
    
    print(">>> Done Processing.")


if __name__ == "__main__":
    preprocess_user_data()