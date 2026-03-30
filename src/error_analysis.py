import torch
from datasets import load_dataset
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from tqdm import tqdm
import pandas as pd
import os
import soundfile as sf

# =========================
# CONFIG
# =========================
SAVE_AUDIO_DIR = "data/error_audio"
OUTPUT_CSV = "data/error_samples.csv"

os.makedirs(SAVE_AUDIO_DIR, exist_ok=True)

# =========================
# LOAD DATA
# =========================
print("Loading dataset...")

dataset = load_dataset(
    "google/fleurs",
    "hi_in",
    split="test[:50]",
    trust_remote_code=True
)

# =========================
# LOAD MODEL
# =========================
print("Loading model...")

processor = WhisperProcessor.from_pretrained("openai/whisper-small")
model = WhisperForConditionalGeneration.from_pretrained("outputs/whisper-small-hi-final")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

# =========================
# ERROR EXTRACTION
# =========================
results = []

print("Extracting errors...")

for i, sample in enumerate(tqdm(dataset)):

    audio = sample["audio"]["array"]
    sr = sample["audio"]["sampling_rate"]

    inputs = processor(audio, sampling_rate=sr, return_tensors="pt")
    input_features = inputs.input_features.to(device)

    with torch.no_grad():
        pred_ids = model.generate(input_features)

    pred_text = processor.batch_decode(pred_ids, skip_special_tokens=True)[0]
    ref_text = sample["transcription"]

    # Only keep errors
    if pred_text.strip() != ref_text.strip():

        # Save audio
        audio_filename = f"error_{i}.wav"
        audio_path = os.path.join(SAVE_AUDIO_DIR, audio_filename)
        sf.write(audio_path, audio, sr)

        results.append({
            "audio": audio_path,
            "reference": ref_text,
            "prediction": pred_text
        })

    if len(results) >= 25:
        break

# =========================
# SAVE CLEAN CSV
# =========================
df = pd.DataFrame(results)

df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

print(f" Saved {len(df)} error samples to {OUTPUT_CSV}")