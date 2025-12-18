import pandas as pd

df = pd.read_csv("data/processed/clean_chat_data.csv")

print("COLUMNS:", df.columns.tolist())
print("\nFIRST 10 ROWS:")
print(df.head(10))

if "label" in df.columns:
    print("\nUNIQUE LABELS:", df["label"].unique())

print("\nTOTAL ROWS:", len(df))