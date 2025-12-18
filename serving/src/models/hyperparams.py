import torch 

class ConfigHyperparams:
    """Central place for paths and hyper-parameters."""

    # Raw/processed data
    RAW_INTERACTION_PATH = "src/resources/raw_items/interaction_final.csv"
    RAW_VIDEO_DIR = "src/resources/raw_items/videos"
    PROCESSED_DIR = "src/resources/data_processed"
    MODEL_SAVE_PATH = "src/resources/models"
    INTERACTION_DATA_DIR = f"{PROCESSED_DIR}/interaction_final.csv"

    # Raw text/video feature inputs (offline extraction)
    TITLE_DIR = "src/resources/raw_items/titles"
    TRANSCRIPT_DIR = "src/resources/raw_items/transcripts"
    VIDEO_INPUT_DIR = "src/resources/raw_items/videos"

    # Preprocess outputs used by training
    TRAIN_PARQUET = f"{PROCESSED_DIR}/train_user.parquet"
    VAL_PARQUET = f"{PROCESSED_DIR}/val_user.parquet"
    VAL_COLD_PARQUET = f"{PROCESSED_DIR}/val_cold.parquet"  # Cold-start users (<=5 interactions)
    VAL_WARM_PARQUET = f"{PROCESSED_DIR}/val_warm.parquet"  # Warm users (>5 interactions)
    MAPPINGS_JSON = f"{PROCESSED_DIR}/user_mappings.json"
    CHECKPOINT_PATH = f"{MODEL_SAVE_PATH}/two_tower_pure_content_ep20.pth"

    # Pre-computed item features
    TEXT_FEATS_PATH = f"{PROCESSED_DIR}/text_features.npy"
    TEXT_IDS_PATH = f"{PROCESSED_DIR}/text_ids.npy"
    VIDEO_FEATS_PATH = f"{PROCESSED_DIR}/video_features.npy"
    VIDEO_IDS_PATH = f"{PROCESSED_DIR}/video_ids.npy"

    # Extraction Params
    TEXT_MODEL_NAME = "BAAI/bge-m3"  # Multilingual model (100+ languages including Chinese)
    TEXT_BATCH_SIZE = 16  # Giảm batch size vì bge-m3 lớn hơn (1024-dim output)
    VIDEO_NUM_FRAMES = 16  # số frame trích xuất từ mỗi video
    VIDEO_IMG_SIZE = 224  # kích thước ảnh đầu vào cho backbone video model

    # Data Params
    MAX_HISTORY = 50  # Lấy 50 video gần nhất
    
    TEXT_DIM = 1024  # BAAI/bge-m3 output dimension (thay vì 768)
    VIDEO_DIM = 768  # Kích thước vector video feature (ViT mean pooling)
    COMBINED_DIM = 1792  # TEXT_DIM (1024) + VIDEO_DIM (768)
    EMBED_DIM = 512  # Final embedding dimension for dot product

    # Model Params
    BERT_MODEL_NAME = "distilbert-base-multilingual-cased"

    # Training Params (Optimized with TransformerEncoder)
    BATCH_SIZE = 64      # Nhỏ hơn cho dataset nhỏ
    LEARNING_RATE = 1e-4  # Higher LR for faster convergence
    WEIGHT_DECAY = 1e-4   # Regularization
    DROPOUT = 0.2         # Reduced dropout (TransformerEncoder has internal dropout)
    EPOCHS = 20           # More epochs for attention to converge
    WARMUP_STEPS = 200    # Longer warmup for TransformerEncoder
    LR = 1e-4  # Consistent with LEARNING_RATE
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4

    # Random Seed
    SEED = 42

    # DEMOGRAPHIC PARAMS (for recommendations)
    AGE_THRESHOLD = 0.5  # Threshold to distinguish 'younger' vs 'older' users (normalized age)