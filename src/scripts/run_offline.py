"""One-shot offline preprocessing entrypoint.

Run this once to create text/video features and parquet splits required by training.
Usage (from repo root):
    python -m src.scripts.run_offline
"""
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.config import Config
from src.scripts.extract_text_features import main as run_text_extract
from src.scripts.extract_video_features import main as run_video_extract
from src.scripts.preprocess import preprocess_user_data
from src.scripts.build_video_metadata import main as build_video_metadata

def _print_check(path_str: str) -> None:
    path = Path(path_str)
    status = "OK" if path.exists() else "MISSING"
    size_mb = path.stat().st_size / 1e6 if path.exists() else 0.0
    print(f"- {path}: {status} ({size_mb:.2f} MB)")


def main() -> None:
    print("=== Offline preprocess: start ===")

    print("\nBuilding video metadata pickle")
    build_video_metadata()
    
    print("-> Extracting text features")
    run_text_extract()

    print("-> Extracting video features")
    run_video_extract()

    print("-> Building user parquet & mappings")
    preprocess_user_data()

    print("\nArtifacts:")
    _print_check(Config.TRAIN_PARQUET)
    _print_check(Config.VAL_PARQUET)
    _print_check(Config.MAPPINGS_JSON)

    print("\nPre-computed item features (expected present):")
    _print_check(Config.TEXT_FEATS_PATH)
    _print_check(Config.TEXT_IDS_PATH)
    _print_check(Config.VIDEO_FEATS_PATH)
    _print_check(Config.VIDEO_IDS_PATH)

    print("=== Offline preprocess: done ===")


if __name__ == "__main__":
    main()
