import pandas as pd

# Load Kaggle spam.csv correctly
df = pd.read_csv(
    "data/raw/chat_data.csv",
    encoding="latin-1"
)

# Keep only required columns
df = df[["v1", "v2"]]
df.columns = ["label", "message"]

# Clean text
df["label"] = df["label"].str.lower().str.strip()
df["message"] = df["message"].astype(str).str.lower().str.strip()

# Keep valid rows
df = df[df["label"].isin(["ham", "spam"])]
df = df[df["message"].str.len() > 0]

df.to_csv("data/processed/clean_chat_data.csv", index=False)

print("✅ Preprocessing successful")
print("Rows:", len(df))
print(df.head())