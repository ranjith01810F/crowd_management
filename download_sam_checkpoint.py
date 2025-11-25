"""Helper script to download SAM checkpoint files."""

import urllib.request
import os
from pathlib import Path

SAM_CHECKPOINTS = {
    "vit_h": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        "filename": "sam_vit_h_4b8939.pth",
        "size": "2.4 GB",
        "description": "ViT-H (Largest, Best Quality)"
    },
    "vit_l": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
        "filename": "sam_vit_l_0b3195.pth",
        "size": "1.2 GB",
        "description": "ViT-L (Medium)"
    },
    "vit_b": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        "filename": "sam_vit_b_01ec64.pth",
        "size": "375 MB",
        "description": "ViT-B (Smallest, Fastest)"
    }
}


def download_file(url: str, filename: str):
    """Download a file with progress bar."""
    def show_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(downloaded * 100 / total_size, 100)
        bar_length = 40
        filled = int(bar_length * downloaded / total_size)
        bar = '=' * filled + '-' * (bar_length - filled)
        print(f'\r[{bar}] {percent:.1f}% ({downloaded / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB)', end='')
    
    print(f"Downloading {filename}...")
    print(f"URL: {url}")
    urllib.request.urlretrieve(url, filename, show_progress)
    print(f"\n✓ Downloaded: {filename}")


def main():
    print("SAM Checkpoint Downloader")
    print("=" * 50)
    print("\nAvailable models:")
    for i, (key, info) in enumerate(SAM_CHECKPOINTS.items(), 1):
        print(f"{i}. {info['description']} ({key}) - {info['size']}")
    
    print("\nRecommended: ViT-B (vit_b) for speed, ViT-H (vit_h) for quality")
    
    choice = input("\nEnter model number (1-3) or name (vit_h/vit_l/vit_b) [default: vit_b]: ").strip().lower()
    
    if not choice:
        choice = "vit_b"
    elif choice.isdigit():
        keys = list(SAM_CHECKPOINTS.keys())
        choice = keys[int(choice) - 1] if 1 <= int(choice) <= 3 else "vit_b"
    
    if choice not in SAM_CHECKPOINTS:
        print(f"Invalid choice. Using default: vit_b")
        choice = "vit_b"
    
    checkpoint = SAM_CHECKPOINTS[choice]
    filename = checkpoint["filename"]
    
    # Check if file already exists
    if Path(filename).exists():
        overwrite = input(f"\n{filename} already exists. Overwrite? (y/n): ").strip().lower()
        if overwrite != 'y':
            print("Cancelled.")
            return
    
    try:
        download_file(checkpoint["url"], filename)
        print(f"\n✓ Success! Checkpoint saved to: {Path(filename).absolute()}")
        print(f"\nTo use this checkpoint, update backend/config.py:")
        print(f'  SAM_CHECKPOINT: str = "{filename}"')
        print(f'  SAM_MODEL_TYPE: str = "{choice}"')
    except Exception as e:
        print(f"\n✗ Error downloading: {e}")
        print("\nManual download:")
        print(f"  URL: {checkpoint['url']}")
        print(f"  Save as: {filename}")


if __name__ == "__main__":
    main()

