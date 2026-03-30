import torch
from datasets import load_dataset
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import evaluate
from tqdm import tqdm

# ==============================
# 1. LOAD FLEURS DATASET
# ==============================
print("Loading FLEURS Hindi dataset...")

dataset = load_dataset(
    "google/fleurs",
    "hi_in",
    split="test[:50]",   # change to "test" for final run
    trust_remote_code=True
)

# ==============================
# 2. LOAD MODELS
# ==============================
print("Loading models...")

processor = WhisperProcessor.from_pretrained("openai/whisper-small")

baseline_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
finetuned_model = WhisperForConditionalGeneration.from_pretrained("outputs/whisper-small-hi-final")

device = "cuda" if torch.cuda.is_available() else "cpu"

baseline_model.to(device)
finetuned_model.to(device)

baseline_model.eval()
finetuned_model.eval()

# ==============================
# 3. LOAD WER METRIC
# ==============================
wer_metric = evaluate.load("wer")

baseline_preds = []
finetuned_preds = []
references = []

# ==============================
# 4. EVALUATION LOOP
# ==============================
print("Evaluating...")

for sample in tqdm(dataset):

    audio = sample["audio"]["array"]
    sr = sample["audio"]["sampling_rate"]

    inputs = processor(audio, sampling_rate=sr, return_tensors="pt")
    input_features = inputs.input_features.to(device)

    with torch.no_grad():
    #  NO forced_decoder_ids (important fix)
        base_ids = baseline_model.generate(input_features)
        fine_ids = finetuned_model.generate(input_features)
    

    base_text = processor.batch_decode(base_ids, skip_special_tokens=True)[0]
    fine_text = processor.batch_decode(fine_ids, skip_special_tokens=True)[0]

    baseline_preds.append(base_text)
    finetuned_preds.append(fine_text)
    references.append(sample["transcription"])

# ==============================
# 5. COMPUTE WER
# ==============================
baseline_wer = wer_metric.compute(
    predictions=baseline_preds,
    references=references
)

finetuned_wer = wer_metric.compute(
    predictions=finetuned_preds,
    references=references
)

# ==============================
# 6. PRINT RESULTS
# ==============================
print("\n===== FLEURS RESULTS =====")
print("Baseline WER:", baseline_wer)
print("Fine-tuned WER:", finetuned_wer)