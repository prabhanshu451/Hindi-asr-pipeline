import pandas as pd
import re

INPUT_CSV = "data/final_dataset.csv"   # your dataset
OUTPUT_CSV = "data/cleaned_output.csv"

df = pd.read_csv(INPUT_CSV)

# ==============================
# 1. NUMBER NORMALIZATION
# ==============================

hindi_numbers = {
    "शून्य": 0, "एक": 1, "दो": 2, "तीन": 3, "चार": 4,
    "पांच": 5, "छह": 6, "सात": 7, "आठ": 8, "नौ": 9,
    "दस": 10, "बीस": 20, "तीस": 30, "चालीस": 40,
    "पचास": 50, "सौ": 100, "हजार": 1000
}

def normalize_numbers(text):
    words = text.split()
    result = []
    skip_next = False

    for i, word in enumerate(words):
        if skip_next:
            skip_next = False
            continue

        if word in hindi_numbers:
            num = hindi_numbers[word]

            # handle compound (e.g., "तीन सौ")
            if i + 1 < len(words) and words[i+1] in ["सौ", "हजार"]:
                num *= hindi_numbers[words[i+1]]
                skip_next = True

            result.append(str(num))
        else:
            result.append(word)

    return " ".join(result)

# ==============================
# 2. ENGLISH WORD DETECTION
# ==============================

def is_english_word(word):
    return re.match(r'^[a-zA-Z]+$', word) is not None

def tag_english_words(text):
    words = text.split()
    tagged = []

    for word in words:
        if is_english_word(word):
            tagged.append(f"[EN]{word}[/EN]")
        else:
            tagged.append(word)

    return " ".join(tagged)

# ==============================
# 3. PIPELINE
# ==============================

def process_text(text):
    text = str(text)

    # step 1: normalize numbers
    text = normalize_numbers(text)

    # step 2: tag english words
    text = tag_english_words(text)

    return text

# ==============================
# APPLY PIPELINE
# ==============================

print("Processing dataset...")

df["cleaned_text"] = df["text"].apply(process_text)

df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

print(f" Saved cleaned output to {OUTPUT_CSV}")