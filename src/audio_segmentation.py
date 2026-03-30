import os
import json
import pandas as pd
import librosa
import soundfile as sf
from tqdm import tqdm

# === PATHS ===
DATA_PATH = "data/FT Data - data.csv"
AUDIO_DIR = "data/audio"
JSON_DIR = "data/transcriptions"
OUTPUT_DIR = "data/segments"

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)

segment_data = []

# === PROCESS EACH RECORDING ===
for idx, row in tqdm(df.iterrows(), total=len(df)):

    recording_id = row['recording_id']

    audio_path = os.path.join(AUDIO_DIR, f"{recording_id}.wav")
    json_path = os.path.join(JSON_DIR, f"{recording_id}.json")

    if not os.path.exists(audio_path) or not os.path.exists(json_path):
        continue

    # Load audio
    try:
        audio, sr = librosa.load(audio_path, sr=16000)
    except:
        continue

    # Load JSON
    with open(json_path, "r", encoding="utf-8") as f:
        segments = json.load(f)

    # === CREATE SEGMENTS ===
    for i, seg in enumerate(segments):

        start = seg.get("start")
        end = seg.get("end")
        text = seg.get("text", "").strip()

        # Skip invalid
        if not text or end <= start:
            continue

        # Convert to samples
        start_sample = int(start * sr)
        end_sample = int(end * sr)

        audio_segment = audio[start_sample:end_sample]

        # Skip very short clips
        if len(audio_segment) < sr:  # <1 sec
            continue

        # Save segment
        segment_filename = f"{recording_id}_{i}.wav"
        segment_path = os.path.join(OUTPUT_DIR, segment_filename)

        sf.write(segment_path, audio_segment, sr)

        segment_data.append({
            "audio": segment_path,
            "text": text
        })

# === SAVE DATASET ===
output_df = pd.DataFrame(segment_data)
output_df.to_csv("data/final_dataset.csv", index=False)

print("\n===== SEGMENTATION DONE =====")
print("Total segments:", len(output_df))