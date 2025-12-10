"""Build video metadata dictionary from interaction CSV for text feature extraction.

This script extracts metadata (categories, tags, duration, author_id) for each video
and saves it to a pickle file for use during text feature extraction.

Usage (from repo root):
    python -m src.scripts.build_video_metadata
"""
import pandas as pd
import pickle
from pathlib import Path
from src.config import Config


def main():
    """Extract video metadata from interaction CSV and save to pickle."""
    
    # Load interaction data
    interaction_path = Config.RAW_INTERACTION_PATH
    print(f"Loading interactions from {interaction_path}...")
    
    if not Path(interaction_path).exists():
        print(f"ERROR: Interaction file not found at {interaction_path}")
        print("Please update Config.RAW_INTERACTION_PATH or run preprocessing first.")
        return
    
    df = pd.read_csv(interaction_path)
    print(f"Loaded {len(df)} interactions for {df['pid'].nunique()} unique videos.")
    
    # Group by video ID and aggregate metadata
    # For each video, take first occurrence of author_id, duration
    # and combine all unique categories/tags
    video_metadata = {}
    
    for pid in df['pid'].unique():
        vid_df = df[df['pid'] == pid]
        
        # Get first values for scalar fields
        first_row = vid_df.iloc[0]
        
        metadata = {
            'author_id': int(first_row.get('author_id', 0)),
            'duration': float(first_row.get('duration', 0.0)),
            'author_fans_count': float(first_row.get('author_fans_count', 0.0)),
        }
        
        # Combine categories if exists (may be string like "Categories: X, Y, Z")
        if 'full_categories' in vid_df.columns:
            categories_set = set()
            for cat_str in vid_df['full_categories'].dropna().unique():
                if isinstance(cat_str, str):
                    # Remove "Categories: " prefix if present
                    cat_str = cat_str.replace("Categories: ", "")
                    categories_set.update([c.strip() for c in cat_str.split(',')])
            metadata['categories'] = ', '.join(sorted(categories_set)) if categories_set else ""
        else:
            metadata['categories'] = ""
        
        # Combine tags if exists
        if 'tag_name' in vid_df.columns:
            tags_set = set()
            for tag_str in vid_df['tag_name'].dropna().unique():
                if isinstance(tag_str, str):
                    # Remove "Tags: " prefix if present
                    tag_str = tag_str.replace("Tags: ", "")
                    tags_set.update([t.strip() for t in tag_str.split(',')])
            metadata['tags'] = ', '.join(sorted(tags_set)) if tags_set else ""
        else:
            metadata['tags'] = ""
        
        video_metadata[str(pid)] = metadata
    
    # Save to pickle
    output_path = Path(Config.PROCESSED_DIR) / "video_metadata.pkl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(video_metadata, f)
    
    print(f"\n✓ Video metadata saved to {output_path}")
    print(f"  Total videos: {len(video_metadata)}")
    
    # Show sample
    sample_id = list(video_metadata.keys())[0]
    print(f"\nSample metadata for video {sample_id}:")
    for k, v in video_metadata[sample_id].items():
        if isinstance(v, str) and len(v) > 100:
            print(f"  {k}: {v[:100]}...")
        else:
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
