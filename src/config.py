import torch


class Config:
    """Central place for paths and hyper-parameters."""

    # Raw/processed data
    RAW_INTERACTION_PATH = "./data/raw_items/interaction_final_cleaned.csv"
    RAW_VIDEO_DIR = "./data/raw_items/videos"
    PROCESSED_DIR = "./data/processed"
    MODEL_SAVE_PATH = "./data/models"

    # Raw text/video feature inputs (offline extraction)
    TITLE_DIR = "./data/raw_items/titles"
    TRANSCRIPT_DIR = "./data/raw_items/transcripts"
    VIDEO_INPUT_DIR = "./data/raw_items/videos"

    # Preprocess outputs used by training
    TRAIN_PARQUET = f"{PROCESSED_DIR}/train_user.parquet"
    VAL_PARQUET = f"{PROCESSED_DIR}/val_user.parquet"
    MAPPINGS_JSON = f"{PROCESSED_DIR}/user_mappings.json"

    # Pre-computed item features
    TEXT_FEATS_PATH = f"{PROCESSED_DIR}/text_features.npy"
    TEXT_IDS_PATH = f"{PROCESSED_DIR}/text_ids.npy"
    VIDEO_FEATS_PATH = f"{PROCESSED_DIR}/video_features.npy"
    VIDEO_IDS_PATH = f"{PROCESSED_DIR}/video_ids.npy"

    # Extraction Params
    TEXT_MODEL_NAME = "paraphrase-albert-small-v2"  # model to extract text features (title + transcript)
    TEXT_BATCH_SIZE = 32  # batch size for text feature extraction
    VIDEO_NUM_FRAMES = 16  # số frame trích xuất từ mỗi video
    VIDEO_IMG_SIZE = 224  # kích thước ảnh đầu vào cho backbone video model

    # Data Params
    MAX_HISTORY = 50  # Lấy 50 video gần nhất

    TEXT_DIM = 768   # Kích thước vector text feature (paraphrase-albert-small-v2)
    VIDEO_DIM = 768  # Kích thước vector video feature (ViT mean pooling)
    COMBINED_DIM = 1536  # TEXT_DIM + VIDEO_DIM (concatenated)
    EMBED_DIM = 1024  # Kích thước vector chung sau projection

    # Model Params
    BERT_MODEL_NAME = "distilbert-base-multilingual-cased"

    # Training Params
    BATCH_SIZE = 256  # Giảm để batchnorm ổn định
    EPOCHS = 30
    LR = 1e-4  # Giảm learning rate để tránh NaN
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4

    # Random Seed
    SEED = 42

    