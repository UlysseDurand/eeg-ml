import rarfile
from tqdm import tqdm
import os

ARCHIVE = 'Data.rar'
TARGET_DIR = 'filtered_data/'
OUT_DIR = 'filtered_data_extracted'

# ensure output dir exists
os.makedirs(OUT_DIR, exist_ok=True)

with rarfile.RarFile(ARCHIVE) as rf:
    members = [m for m in rf.infolist() if m.filename.startswith(TARGET_DIR)]
    for m in tqdm(members, desc='Extracting filtered_data', unit='file'):
        rf.extract(m, path=OUT_DIR)

