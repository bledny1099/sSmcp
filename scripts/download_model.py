#!/usr/bin/env python3
"""Pre-download Faster-Whisper model weights into the cache directory."""

import os
import sys
from faster_whisper import WhisperModel


def main() -> None:
    model_size = sys.argv[1] if len(sys.argv) > 1 else os.getenv("SSMCP_WHISPER_MODEL", "small")
    cache_dir = os.getenv(
        "SSMCP_WHISPER_CACHE_DIR",
        os.path.join(os.path.expanduser("~"), ".cache", "ssmcp", "whisper"),
    )
    os.makedirs(cache_dir, exist_ok=True)
    print(f"Downloading Faster-Whisper '{model_size}' into {cache_dir}...")
    WhisperModel(model_size, device="cpu", compute_type="int8", download_root=cache_dir)
    print("Model download complete.")


if __name__ == "__main__":
    main()
