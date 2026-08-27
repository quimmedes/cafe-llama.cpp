import os
import sys
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from huggingface_hub import hf_hub_download

repo_id = "Qwen/Qwen3.8-Flash-Next"
local_dir = r"H:\Qwen3.8-Flash-Next"
os.makedirs(local_dir, exist_ok=True)

# Fetch file list via API
api_url = f"https://huggingface.co/api/models/{repo_id}"
req = urllib.request.Request(api_url, headers={"User-Agent": "HF-Downloader"})
data = json.loads(urllib.request.urlopen(req).read().decode("utf-8"))

files_to_download = [f["rfilename"] for f in data.get("siblings", []) if not f["rfilename"].startswith(".")]
print(f"Total files to check/download: {len(files_to_download)}")

def download_one(filename):
    dest = os.path.join(local_dir, filename)
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        # Quick check if it's not a small stub
        if not filename.endswith(".safetensors") or os.path.getsize(dest) > 10 * 1024 * 1024:
            return filename, "skipped (exists)"
    
    for attempt in range(5):
        try:
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=local_dir,
                local_dir_use_symlinks=False
            )
            return filename, "downloaded"
        except Exception as e:
            if attempt == 4:
                return filename, f"error: {e}"
            time.sleep(2)

print("Starting parallel file download...")
completed = 0
with ThreadPoolExecutor(max_workers=4) as executor:
    futures = {executor.submit(download_one, f): f for f in files_to_download}
    for future in as_completed(futures):
        fname, status = future.result()
        completed += 1
        print(f"[{completed}/{len(files_to_download)}] {fname}: {status}", flush=True)

print("All files processed!")
