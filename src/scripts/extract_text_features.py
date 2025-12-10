"""Extract text embeddings combining title, transcript, categories, and tags using SentenceTransformer.

Usage (from repo root):
    python -m src.scripts.extract_text_features
"""
import os
import re
import pickle
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

from src.config import Config


def clean_text(text: str, remove_non_latin: bool = False) -> str:
    """Clean and normalize text.
    
    Args:
        text: Input text
        remove_non_latin: If True, remove non-Latin characters (Chinese, etc.)
                         If False, keep all unicode characters
    """
    text = text.lower()
    
    if remove_non_latin:
        # Remove non-Latin characters (for English-only text)
        text = re.sub(r'[^a-z\s]', ' ', text)
    else:
        # Keep unicode (Chinese, etc.), only remove special symbols
        text = re.sub(r'[^\w\s]', ' ', text, flags=re.UNICODE)
    
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def get_processed_text_for_id(
    video_id: str, 
    title_dir: Path, 
    transcript_dir: Path, 
    video_metadata: dict,
    preprocess: bool = True
) -> tuple[str, str]:
    """Load and process text for a video, separating English and Chinese content.
    
    Returns:
        (english_text, chinese_text): Tuple of processed English and Chinese text
    """
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

    # Get metadata from video_metadata dict
    metadata = video_metadata.get(video_id, {})
    categories = metadata.get('categories', '')
    tags = metadata.get('tags', '')

    # Separate English and Chinese content
    # English: title + script + categories (all in English)
    english_text = (
        '[TITLE] ' + title + 
        " [SCRIPT] " + script + 
        " [CATEGORIES] " + categories
    )
    
    # Chinese: tags (often in Chinese)
    chinese_text = '[TAGS] ' + tags

    if preprocess:
        # Clean English text (remove non-Latin for consistency)
        english_text = clean_text(english_text, remove_non_latin=True)
        # Clean Chinese text (keep unicode characters)
        chinese_text = clean_text(chinese_text, remove_non_latin=False)
    
    return english_text, chinese_text


def main() -> None:
    video_dir = Path(Config.VIDEO_INPUT_DIR)
    title_dir = Path(Config.TITLE_DIR)
    transcript_dir = Path(Config.TRANSCRIPT_DIR)
    metadata_path = Path(Config.PROCESSED_DIR) / "video_metadata.pkl"

    assert video_dir.exists(), f"Video dir not found: {video_dir}"
    assert title_dir.exists(), f"Title dir not found: {title_dir}"
    assert transcript_dir.exists(), f"Transcript dir not found: {transcript_dir}"
    
    # Load video metadata (categories, tags)
    if metadata_path.exists():
        print(f"Loading video metadata from {metadata_path}...")
        with open(metadata_path, 'rb') as f:
            video_metadata = pickle.load(f)
        print(f"Loaded metadata for {len(video_metadata)} videos.")
    else:
        print(f"WARNING: Video metadata not found at {metadata_path}")
        print("Run 'python -m src.scripts.build_video_metadata' first to extract metadata.")
        print("Proceeding without metadata (categories and tags will be empty)...\n")
        video_metadata = {}

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

    # Separate English and Chinese content
    all_english_content = []
    all_chinese_content = []
    
    for video_id in tqdm(video_ids, desc="Reading text"):
        english_text, chinese_text = get_processed_text_for_id(
            video_id, title_dir, transcript_dir, video_metadata, preprocess=True
        )
        all_english_content.append(english_text if english_text.strip() else "")
        all_chinese_content.append(chinese_text if chinese_text.strip() else "")

    print("Text content prepared. Starting batch embedding process...")

    # Embed English content
    print("Encoding English content (title + script + categories)...")
    english_feats = model.encode(
        all_english_content,
        batch_size=Config.TEXT_BATCH_SIZE,
        show_progress_bar=True
    ).astype(np.float32)

    # Embed Chinese content  
    print("Encoding Chinese content (tags)...")
    chinese_feats = model.encode(
        all_chinese_content,
        batch_size=Config.TEXT_BATCH_SIZE,
        show_progress_bar=True
    ).astype(np.float32)

    # Combine: Weighted average (80% English, 20% Chinese)
    # This keeps dimension at 1024 while giving proper weight to both languages
    print("Combining English and Chinese embeddings...")
    feats = 0.8 * english_feats + 0.2 * chinese_feats

    os.makedirs(Path(Config.TEXT_FEATS_PATH).parent, exist_ok=True)
    np.save(Config.TEXT_FEATS_PATH, feats)
    np.save(Config.TEXT_IDS_PATH, np.array(video_ids))

    print(f"\nText embeddings successfully saved to '{Config.TEXT_FEATS_PATH}'")
    print(f"Shape of the saved embeddings array: {feats.shape}")
    print(f"IDs saved to {Config.TEXT_IDS_PATH}, count={len(video_ids)}")


if __name__ == "__main__":
    main()
