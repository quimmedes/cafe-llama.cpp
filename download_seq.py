import os
import sys
import time
import json
import urllib.request
from huggingface_hub import hf_hub_download

repo_id = "Qwen/Qwen3.8-Flash-Next"
local_dir = r"H:\Qwen3.8-Flash-Next"
os.makedirs(local_dir, exist_ok=True)

# Fetch file list
api_url = f"https://huggingface.co/api/models/{repo_id}"
req = urllib.request.Request(api_url, headers={"User-Agent": "HF-Downloader"})
data = json.loads(urllib.request.urlopen(req).read().decode("utf-8"))

files_to_download = [f["rfilename"] for f in data.get("siblings", []) if not f["rfilename"].startswith(".")]
print(f"Total files: {len(files_to_download)}", flush=True)

for i, filename in enumerate(files_to_download, 1):
    dest = os.path.join(local_dir, filename)
    if os.path.exists(dest) and os.path.getsize(dest) > 1024:
        print(f"[{i}/{len(files_to_download)}] Exists: {filename} ({os.path.getsize(dest)/(1024*1024):.1f} MB)", flush=True)
        continue
    
    print(f"[{i}/{len(files_to_download)}] Downloading: {filename}...", flush=True)
    for attempt in range(5):
        try:
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=local_dir,
                local_dir_use_symlinks=False
            )
            print(f"[{i}/{len(files_to_download)}] Done: {filename}", flush=True)
            break
        except Exception as e:
            print(f"[{i}/{len(files_to_download)}] Retry {attempt+1} on {filename}: {e}", flush=True)
            time.sleep(3)

print("All downloads finished!", flush=True)
