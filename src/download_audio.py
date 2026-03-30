import pandas as pd
import requests
import os
from tqdm import tqdm

DATA_PATH = "data/FT Data - data.csv"
SAVE_DIR = "data/audio"

os.makedirs(SAVE_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)

success_count = 0
fail_count = 0

for idx, row in tqdm(df.iterrows(), total=len(df)):

    old_url = row['rec_url_gcp']
    
    #  SAME FIX AS BEFORE
    new_url = old_url.replace(
        "joshtalks-data-collection/hq_data/hi",
        "upload_goai"
    )

    try:
        response = requests.get(new_url, timeout=15)

        if response.status_code == 200:
            recording_id = row['recording_id']
            save_path = os.path.join(SAVE_DIR, f"{recording_id}.wav")

            with open(save_path, "wb") as f:
                f.write(response.content)

            success_count += 1
        else:
            print(f" Failed: {new_url} | Status: {response.status_code}")
            fail_count += 1

    except Exception as e:
        print(f" Error: {new_url} | {e}")
        fail_count += 1

print("\n===== AUDIO DOWNLOAD SUMMARY =====")
print("Success:", success_count)
print("Failed:", fail_count)