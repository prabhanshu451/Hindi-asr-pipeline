import pandas as pd
import librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
from jiwer import wer

# ==============================
# LOAD DATA (small sample)
# ==============================
df = pd.read_csv("data/final_dataset.csv")

# take small sample (important)
df = df.sample(50, random_state=42)

# ==============================
# LOAD MODELS
# ==============================
baseline_processor = WhisperProcessor.from_pretrained("openai/whisper-small")
baseline_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")

fine_processor = WhisperProcessor.from_pretrained("outputs/whisper-small-hi-final")
fine_model = WhisperForConditionalGeneration.from_pretrained("outputs/whisper-small-hi-final")

device = "cuda" if torch.cuda.is_available() else "cpu"
baseline_model.to(device)
fine_model.to(device)

# ==============================
# PREDICTION FUNCTION
# ==============================
def predict(model, processor, audio_path):
    audio, sr = librosa.load(audio_path, sr=16000)

    inputs = processor(audio, sampling_rate=sr, return_tensors="pt")
    input_features = inputs.input_features.to(device)

    predicted_ids = model.generate(input_features)

    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

    return transcription

# ==============================
# RUN EVALUATION
# ==============================
baseline_preds = []
fine_preds = []
refs = []

for _, row in df.iterrows():
    audio_path = row["audio"]
    text = row["text"]

    try:
        base_pred = predict(baseline_model, baseline_processor, audio_path)
        fine_pred = predict(fine_model, fine_processor, audio_path)

        baseline_preds.append(base_pred)
        fine_preds.append(fine_pred)
        refs.append(text)

    except:
        continue

# ==============================
# COMPUTE WER
# ==============================
baseline_wer = wer(refs, baseline_preds)
fine_wer = wer(refs, fine_preds)

print("\n===== RESULTS =====")
print("Baseline WER:", baseline_wer)
print("Fine-tuned WER:", fine_wer)