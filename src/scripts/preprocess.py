import json
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tqdm import tqdm

from src.config import Config


def preprocess_user_data():
    """Create user-side parquet + mappings for the two-tower training step."""
    os.makedirs(Config.PROCESSED_DIR, exist_ok=True)

    print(">>> Loading Data...")
    df = pd.read_csv(Config.RAW_INTERACTION_PATH)
    
    # 1. Xử lý Thời gian (Context)
    df['dt'] = pd.to_datetime(df['exposed_time'], unit='s') # Hoặc format phù hợp
    df['p_dow'] = df['dt'].dt.dayofweek  # 0: Mon, 6: Sun (Thay cho p_date)
    # p_hour đã có sẵn
    
    # 2. Xử lý Numerical Features (Age, Price)
    # Fill NaN bằng median hoặc -1
    df['age'] = df['age'].fillna(df['age'].median())
    df['mod_price'] = df['mod_price'].fillna(0)
    
    scaler = MinMaxScaler()
    df[['age', 'mod_price']] = scaler.fit_transform(df[['age', 'mod_price']])
    
    # 3. Xử lý Categorical Features (City, Gender...)
    # Ta dùng LabelEncoder để chuyển string -> int ID
    cat_cols = ['user_id', 'pid', 'gender', 'fre_city', 'fre_community_type', 'fre_city_level']
    mappings = {}
    
    for col in cat_cols:
        df[col] = df[col].fillna("Unknown").astype(str)
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        # Lưu mapping để dùng lúc inference
        mappings[col] = {str(k): int(v) for k, v in zip(le.classes_, le.transform(le.classes_))}
        # +1 vì ta sẽ dành số 0 cho Padding/Unknown sau này
        df[col] = df[col] + 1 
        
    # Lưu file mapping
    with open(Config.MAPPINGS_JSON, "w") as f:
        json.dump(mappings, f)
        
    # 4. Tạo Nhãn (Label) & History Sequence
    # Sắp xếp theo User và Thời gian
    df = df.sort_values(['user_id', 'exposed_time'])
    
    # Logic tạo nhãn (Target)
    # Ví dụ: Xem > 30% duration HOẶC like/share/comment
    df['watch_ratio'] = df['watch_time'] / (df['duration'] + 1e-5)
    df['target'] = (
        (df['watch_ratio'] >= 0.3) | 
        (df['cvm_like'] == 1) | 
        (df['collect'] == 1) |
        (df['forward'] == 1)
    ).astype(int)

    # Logic tạo History: Cửa sổ trượt (Sliding Window)
    # Đây là bước tốn thời gian nhất. Với data lớn dùng Spark/Pandas Window.
    print(">>> Generating User History Sequence...")
    
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
    
    # Padding history để lưu được vào parquet/numpy
    # (Thực tế nên làm trong __getitem__ của Dataset class, nhưng làm ở đây cho dễ debug)
    
    # 5. Lưu kết quả
    # Chỉ giữ lại cột cần thiết cho User Tower và Training
    final_cols = [
        'user_id', 'pid', 'target', 
        'gender', 'age', 'mod_price', 
        'fre_city', 'fre_community_type', 'fre_city_level',
        'p_dow', 'p_hour',
        'history_pids' 
    ]
    
    # Chia train/val theo thời gian (80-20)
    split_idx = int(len(df) * 0.9)
    train_df = df.iloc[:split_idx][final_cols]
    val_df = df.iloc[split_idx:][final_cols]
    
    train_df.to_parquet(Config.TRAIN_PARQUET)
    val_df.to_parquet(Config.VAL_PARQUET)
    print(">>> Done Processing.")

if __name__ == "__main__":
    preprocess_user_data()