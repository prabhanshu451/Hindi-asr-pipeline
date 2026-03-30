import pandas as pd

INPUT_CSV = "data/error_samples.csv"
OUTPUT_CSV = "data/error_categorized.csv"

df = pd.read_csv(INPUT_CSV)

def categorize_error(ref, pred):
    ref_words = ref.split()
    pred_words = pred.split()

    if len(pred_words) < len(ref_words):
        return "Deletion"
    elif len(pred_words) > len(ref_words):
        return "Insertion"
    else:
        if ref != pred:
            return "Substitution"
        else:
            return "Correct"

df["error_type"] = df.apply(
    lambda x: categorize_error(str(x["reference"]), str(x["prediction"])),
    axis=1
)

df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

print(" Error categorization saved to:", OUTPUT_CSV)