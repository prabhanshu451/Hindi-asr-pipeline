import pandas as pd
import librosa
from datasets import Dataset
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    TrainingArguments,
    Trainer
)
import torch

# ==============================
# 1. LOAD DATA
# ==============================
print("Loading dataset...")

df = pd.read_csv("data/final_dataset.csv")
dataset = Dataset.from_pandas(df)

print("Dataset loaded:", dataset)

# ==============================
# 2. LOAD MODEL
# ==============================
print("Loading Whisper model...")

processor = WhisperProcessor.from_pretrained("openai/whisper-small")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")

#  Force Hindi transcription
forced_decoder_ids = processor.get_decoder_prompt_ids(
    language="hi",
    task="transcribe"
)
model.config.forced_decoder_ids = forced_decoder_ids

print("Model loaded successfully")

# ==============================
# 3. PREPROCESS FUNCTION
# ==============================
def preprocess(batch):
    audio_path = batch["audio"]

    audio, sr = librosa.load(audio_path, sr=16000)

    inputs = processor(
        audio,
        sampling_rate=sr,
        text=batch["text"]
    )

    batch["input_features"] = inputs.input_features[0]
    batch["labels"] = inputs.labels

    return batch

print("Processing dataset...")

dataset = dataset.map(preprocess)

# Remove unnecessary columns
dataset = dataset.remove_columns(["audio", "text"])

print("Dataset preprocessing complete")

# ==============================
# 4. CUSTOM DATA COLLATOR  (IMPORTANT)
# ==============================
def data_collator(features):
    input_features = [f["input_features"] for f in features]
    labels = [f["labels"] for f in features]

    # Pad audio inputs
    batch = processor.feature_extractor.pad(
        {"input_features": input_features},
        return_tensors="pt"
    )

    # Pad labels
    labels_batch = processor.tokenizer.pad(
        {"input_ids": labels},
        return_tensors="pt"
    )

    # Replace padding with -100 (ignore in loss)
    labels = labels_batch["input_ids"].masked_fill(
        labels_batch.attention_mask.ne(1), -100
    )

    batch["labels"] = labels

    return batch

# ==============================
# 5. TRAINING CONFIG
# ==============================
training_args = TrainingArguments(
    output_dir="outputs/whisper-small-hi",

    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,

    learning_rate=1e-5,
    warmup_steps=100,
    max_steps=1000,

    fp16=torch.cuda.is_available(),

    logging_steps=50,
    save_steps=500,

    save_total_limit=2,
    report_to="none"
)

# ==============================
# 6. TRAINER
# ==============================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator,   #  our custom collator
)

# ==============================
# 7. TRAIN
# ==============================
print(" Training started...")

trainer.train()

print(" Training completed!")

# ==============================
# 8. SAVE MODEL
# ==============================
model.save_pretrained("outputs/whisper-small-hi-final")
processor.save_pretrained("outputs/whisper-small-hi-final")

print(" Model saved!")