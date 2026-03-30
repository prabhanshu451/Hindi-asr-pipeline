import pandas as pd
import requests

file_path = "data/FT Data - data.csv"
df = pd.read_csv(file_path)

print("\n=== URL VALIDATION ===")

sample_df = df.head(3)

for i, row in sample_df.iterrows():
    print(f"\nRow {i}")
    
    audio_url = row['rec_url_gcp']
    transcription_url = row['transcription_url_gcp']
    
    try:
        audio_status = requests.get(audio_url, timeout=10).status_code
    except Exception as e:
        audio_status = f"ERROR: {e}"
        
    try:
        transcription_status = requests.get(transcription_url, timeout=10).status_code
    except Exception as e:
        transcription_status = f"ERROR: {e}"
    
    print("Audio URL:", audio_url)
    print("Status:", audio_status)
    
    print("Transcription URL:", transcription_url)
    print("Status:", transcription_status)