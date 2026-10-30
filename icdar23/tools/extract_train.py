#!/usr/bin/env python3
import os
import tarfile
from pathlib import Path

def extract_tar(tar_path: str, out_dir: str) -> Path:
    tar_path = Path(tar_path).expanduser().resolve()
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Quick skip if it looks already extracted (you can adjust this heuristic)
    marker = out_dir / ".extracted_ok"
    if marker.exists():
        return out_dir

    if not tar_path.exists():
        raise FileNotFoundError(f"Tar not found: {tar_path}")

    # Extract
    with tarfile.open(tar_path, "r:*") as tf:
        tf.extractall(out_dir)

    marker.write_text(str(tar_path) + "\n")
    return out_dir

if __name__ == "__main__":
    # Example: set these via environment variables in sbatch
    tar_path = os.environ["TRAIN_TAR"]          # e.g. /gpfs/.../icdar2017_train_patches.tar
    out_dir  = os.environ["TRAIN_EXTRACT_DIR"]  # e.g. /local/.../patches/train

    extracted = extract_tar(tar_path, out_dir)
    print(f"Extracted to: {extracted}")