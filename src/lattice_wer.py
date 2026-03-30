import pandas as pd

INPUT_CSV = "data/Question 4 - Task.csv"
OUTPUT_CSV = "data/lattice_wer_output.csv"


# ---------------------------
# STEP 1: Build lattice
# ---------------------------
def build_lattice(models_outputs):
    max_len = max(len(m) for m in models_outputs)

    lattice = []

    for i in range(max_len):
        words_at_pos = set()

        for model in models_outputs:
            if i < len(model):
                words_at_pos.add(model[i])

        lattice.append(list(words_at_pos))

    return lattice


# ---------------------------
# STEP 2: Smart matching
# ---------------------------
def normalize_number(word):
    mapping = {
        "14": ["चौदह"],
        "2": ["दो"],
        "10": ["दस"]
    }
    return mapping.get(word, [])


def match_word(ref_word, lattice_words):
    # direct match
    if ref_word in lattice_words:
        return True

    # number normalization
    for alt in normalize_number(ref_word):
        if alt in lattice_words:
            return True

    return False


# ---------------------------
# STEP 3: Lattice WER
# ---------------------------
def lattice_wer(reference, lattice):

    errors = 0
    total = len(reference)

    for i in range(total):

        if i >= len(lattice):
            errors += 1
            continue

        ref_word = reference[i]
        lattice_words = lattice[i]

        if not match_word(ref_word, lattice_words):
            errors += 1

    return errors / total if total > 0 else 0


# ---------------------------
# STEP 4: MAIN
# ---------------------------
df = pd.read_csv(INPUT_CSV)

# Remove garbage column
df = df.drop(columns=["Unnamed: 8"], errors="ignore")

results = []

model_cols = [
    "Model H", "Model i", "Model k",
    "Model l", "Model m", "Model n"
]

for idx, row in df.iterrows():

    # Reference (Human transcription)
    ref = str(row["Human"]).split()

    # Model outputs
    model_outputs = [
        str(row[col]).split()
        for col in model_cols
        if col in df.columns
    ]

    # Build lattice
    lattice = build_lattice(model_outputs)

    # Compute WER
    wer = lattice_wer(ref, lattice)

    results.append(wer)

# Save results
df["lattice_wer"] = results
df.to_csv(OUTPUT_CSV, index=False)

print(" Lattice WER computed successfully")
print(" Saved to:", OUTPUT_CSV)