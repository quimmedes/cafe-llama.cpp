import os
import sys
import time
from huggingface_hub import snapshot_download

local_dir = r"H:\Qwen3.8-Flash-Next"
os.makedirs(local_dir, exist_ok=True)

print(f"Starting/resuming download of Qwen/Qwen3.8-Flash-Next to {local_dir}...")
max_retries = 10
for attempt in range(1, max_retries + 1):
    try:
        path = snapshot_download(
            repo_id="Qwen/Qwen3.8-Flash-Next",
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            resume_download=True,
            max_workers=4
        )
        print("Download completed successfully at:", path)
        break
    except Exception as e:
        print(f"Attempt {attempt} failed: {e}. Retrying in 5 seconds...", file=sys.stderr)
        time.sleep(5)
