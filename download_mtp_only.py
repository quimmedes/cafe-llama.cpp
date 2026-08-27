import os
import sys
import time
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from huggingface_hub import hf_hub_download

repo_id = "Qwen/Qwen3.8-Flash-Next"
local_dir = r"H:\Qwen3.8-Flash-Next"
os.makedirs(local_dir, exist_ok=True)

# Fetch index
print("Fetching model index from Hugging Face...")
index_url = f"https://huggingface.co/{repo_id}/raw/main/model.safetensors.index.json"
idx_data = json.loads(urllib.request.urlopen(index_url).read().decode("utf-8"))

# Identify MTP and shared shards
weight_map = idx_data["weight_map"]
mtp_shards = set(v for k, v in weight_map.items() if "mtp" in k.lower())
shared_shards = set(v for k, v in weight_map.items() if any(s in k for s in ["embed_tokens", "model.norm", "lm_head"]))
target_shards = sorted(list(mtp_shards.union(shared_shards)))

# Essential config and tokenizer files
config_files = [
    "config.json", "generation_config.json", "model.safetensors.index.json",
    "tokenizer.json", "tokenizer_config.json", "merges.txt", "vocab.json", "chat_template.jinja"
]

all_files = config_files + target_shards
print(f"Targeting {len(target_shards)} MTP shards + {len(config_files)} config files ({len(all_files)} total).", flush=True)

def download_file(filename):
    dest = os.path.join(local_dir, filename)
    if os.path.exists(dest) and os.path.getsize(dest) > 1024:
        # Check if complete (at least 10MB for safetensors)
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
                return filename, f"failed: {e}"
            time.sleep(2)

print("Starting parallel download of MTP files (4 workers)...", flush=True)
completed = 0
with ThreadPoolExecutor(max_workers=4) as executor:
    futures = {executor.submit(download_file, f): f for f in all_files}
    for future in as_completed(futures):
        fname, status = future.result()
        completed += 1
        print(f"[{completed}/{len(all_files)}] {fname}: {status}", flush=True)

print("MTP files download complete!", flush=True)
