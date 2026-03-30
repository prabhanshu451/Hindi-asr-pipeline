import pandas as pd
import re

#  USE ASSIGNMENT FILE ONLY
INPUT_CSV = "data/Unique Words Data - Sheet1.csv"
OUTPUT_CSV = "data/word_analysis.csv"

df = pd.read_csv(INPUT_CSV)

# Ensure correct column name
df.columns = ["word"]

# ==============================
# VALIDATION
# ==============================
def is_valid_hindi(word):
    return bool(re.match(r'^[\u0900-\u097F]+$', str(word)))


# ==============================
# COMMON WORDS
# ==============================
COMMON_WORDS = {
    "है", "हैं", "मैं", "तुम", "आप", "हम", "और", "का", "की", "के",
    "यह", "वह", "था", "थी", "थे", "से", "में", "पर", "को", "नहीं"
}


# ==============================
# CLASSIFICATION
# ==============================
def classify_word(word):
    word = str(word).strip()

    # Common words
    if word in COMMON_WORDS:
        return "correct", "high", "common Hindi word"

    # Valid Hindi script
    if is_valid_hindi(word):

        # repetition error (ASR noise)
        if any(word.count(c) > 3 for c in set(word)):
            return "incorrect", "high", "repetition error (ASR)"

        # very short words
        if len(word) <= 2:
            return "correct", "high", "valid short word"

        # rare but valid words
        if len(word) > 2 and len(word) <= 12:
            return "correct", "medium", "valid but possibly rare word"

        # too long → likely error
        if len(word) > 15:
            return "incorrect", "medium", "unnaturally long word"

    # English / mixed words
    if re.search(r'[a-zA-Z]', word):
        return "incorrect", "medium", "mixed language word"

    # fallback
    return "incorrect", "low", "unknown or noisy word"


# Apply
df[["label", "confidence", "reason"]] = df["word"].apply(
    lambda x: pd.Series(classify_word(x))
)

# Save
df.to_csv(OUTPUT_CSV, index=False)

print(f" Saved results to {OUTPUT_CSV}")