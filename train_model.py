import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score

# Load cleaned data
df = pd.read_csv("data/processed/clean_chat_data.csv")

# Remove empty rows
df = df.dropna(subset=["message", "label"])
df = df[df["message"].str.strip() != ""]

# Encode labels
df["label"] = df["label"].map({"ham": 0, "spam": 1})

X = df["message"]
y = df["label"]

print("TOTAL SAMPLES:", len(X))

# Improved TF-IDF
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    stop_words="english",
    sublinear_tf=True
)

X_vec = vectorizer.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42, stratify=y
)

# Strong classifier
svm = LinearSVC(class_weight="balanced")

# Calibrate for probabilities
model = CalibratedClassifierCV(svm)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"🎯 FINAL ACCURACY: {accuracy * 100:.2f}%")

# Save model & vectorizer
with open("model/model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("model/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("✅ High-accuracy model saved")