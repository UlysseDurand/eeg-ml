#!/usr/bin/env python3
import os
import sys
import requests
from tqdm import tqdm

API_BASE   = "https://api.figshare.com/v2"
ARTICLE_ID = "14562090"
TOKEN      = os.getenv("FIGSHARE_TOKEN")  # optional for private files
HEADERS    = {"Authorization": f"token {TOKEN}"} if TOKEN else {}
DEST_DIR   = sys.argv[1] if len(sys.argv) > 1 else "."

def download_file(url: str, path: str):
    """Download a file with a tqdm progress bar."""
    resp = requests.get(url, stream=True)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    with open(path, "wb") as f, tqdm(
        total=total,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        desc=os.path.basename(path),
    ) as bar:
        for chunk in resp.iter_content(chunk_size=8192):
            if not chunk:
                continue
            f.write(chunk)
            bar.update(len(chunk))  # update by bytes written :contentReference[oaicite:0]{index=0}

def main():
    os.makedirs(DEST_DIR, exist_ok=True)
    # 1) List all files for this article
    resp = requests.get(f"{API_BASE}/articles/{ARTICLE_ID}/files", headers=HEADERS)
    resp.raise_for_status()
    files = resp.json()

    # 2) Download each file with progress
    for file_meta in files:
        name = file_meta.get("name", str(file_meta["id"]))
        dl   = file_meta["download_url"]
        out  = os.path.join(DEST_DIR, name)
        print(f"→ Downloading {name} …")
        download_file(dl, out)
        print(f"✔ Saved → {out}")

if __name__ == "__main__":
    main()

