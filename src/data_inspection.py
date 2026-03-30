import pandas as pd

# Load dataset
file_path = "data/FT Data - data.csv"
df = pd.read_csv(file_path)

print("File Loaded:", file_path)
print("Shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

print("\nMissing Values:")
print(df.isnull().sum())