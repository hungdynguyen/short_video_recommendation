"""Extract text embeddings combining title and transcript using SentenceTransformer.

Usage (from repo root):
    python -m src.scripts.extract_text_features
"""
import os
import re
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

from src.config import Config


def clean_text(text: str) -> str:
    """Clean and normalize text."""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def get_processed_text_for_id(video_id: str, title_dir: Path, transcript_dir: Path, preprocess: bool = True) -> str:
    """Load and combine title and transcript for a given video ID."""
    title_path = title_dir / f"{video_id}.txt"
    script_path = transcript_dir / f"{video_id}.txt"

    try:
        with title_path.open('r', encoding='utf-8') as f:
            title = f.read()
    except FileNotFoundError:
        print(f"Warning: No title file found for ID {video_id}")
        title = ""

    try:
        with script_path.open('r', encoding='utf-8') as f:
            script = f.read()
    except FileNotFoundError:
        print(f"Warning: No script file found for ID {video_id}")
        script = ""

    combined_text = '[TITLE] ' + title + " [SCRIPT] " + script

    if preprocess:
        return clean_text(combined_text)
    else:
        return combined_text


def main() -> None:
    video_dir = Path(Config.VIDEO_INPUT_DIR)
    title_dir = Path(Config.TITLE_DIR)
    transcript_dir = Path(Config.TRANSCRIPT_DIR)

    assert video_dir.exists(), f"Video dir not found: {video_dir}"
    assert title_dir.exists(), f"Title dir not found: {title_dir}"
    assert transcript_dir.exists(), f"Transcript dir not found: {transcript_dir}"

    # Get video IDs from video files, sorted numerically
    video_files = sorted([f for f in video_dir.iterdir() if f.suffix.lower() == '.mp4'],
                         key=lambda x: int(x.stem))
    video_ids = [f.stem for f in video_files]

    print(f"Found {len(video_ids)} videos. Preparing text content...")

    if not video_ids:
        print("No videos found to process. Exiting.")
        return

    print(f"Loading text embedding model ({Config.TEXT_MODEL_NAME})...")
    device = "cuda" if __import__('torch').cuda.is_available() else "cpu"
    model = SentenceTransformer(Config.TEXT_MODEL_NAME, device=device)
    print("Text embedding model loaded.")

    all_text_content = []
    for video_id in tqdm(video_ids, desc="Reading text"):
        text_content = get_processed_text_for_id(video_id, title_dir, transcript_dir, preprocess=True)
        if not text_content.strip():
            all_text_content.append("")
        else:
            all_text_content.append(text_content)

    print("Text content prepared. Starting batch embedding process...")

    feats = model.encode(
        all_text_content,
        batch_size=Config.TEXT_BATCH_SIZE,
        show_progress_bar=True
    ).astype(np.float32)

    os.makedirs(Path(Config.TEXT_FEATS_PATH).parent, exist_ok=True)
    np.save(Config.TEXT_FEATS_PATH, feats)
    np.save(Config.TEXT_IDS_PATH, np.array(video_ids))

    print(f"\nText embeddings successfully saved to '{Config.TEXT_FEATS_PATH}'")
    print(f"Shape of the saved embeddings array: {feats.shape}")
    print(f"IDs saved to {Config.TEXT_IDS_PATH}, count={len(video_ids)}")


if __name__ == "__main__":
    main()
