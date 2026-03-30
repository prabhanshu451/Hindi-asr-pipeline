import pandas as pd

df = pd.read_csv("data/word_analysis.csv")

correct_count = (df["label"] == "correct").sum()

print("Total correct words:", correct_count)