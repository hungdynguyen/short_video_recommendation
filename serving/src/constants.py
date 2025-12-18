# Configuration file paths
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


CONFIG_PATH_PIPELINE = BASE_DIR / "resources" / "config_pipeline.yml"
CONFIG_VIDEO_METADATA_FILE = BASE_DIR / "resources" / "data_processed" / "video_metadata.pkl"
