import requests
from tqdm import tqdm

ARTICLE_ID = 14562090
FILE_ID    = 27956376

# 1. Fetch file metadata
files = requests.get(f'https://api.figshare.com/v2/articles/{ARTICLE_ID}/files').json()
fmeta = next(f for f in files if f['id'] == FILE_ID)

# 2. Download with progress bar
url      = fmeta['download_url']
fname    = fmeta['name']
resp     = requests.get(url, stream=True)
total    = int(resp.headers.get('content-length', 0))
chunk_sz = 1024

with open(fname, 'wb') as f, \
     tqdm(total=total, unit='B', unit_scale=True, desc=fname) as bar:
    for chunk in resp.iter_content(chunk_sz):
        f.write(chunk)
        bar.update(len(chunk))

