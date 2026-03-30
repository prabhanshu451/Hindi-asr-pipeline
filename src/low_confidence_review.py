import pandas as pd

INPUT_CSV = "data/word_analysis.csv"
OUTPUT_CSV = "data/low_confidence_sample.csv"

df = pd.read_csv(INPUT_CSV)

low_conf = df[df["confidence"] == "low"]

sample = low_conf.sample(min(40, len(low_conf)))

sample.to_csv(OUTPUT_CSV, index=False)

print(f" Saved {len(sample)} low confidence words")