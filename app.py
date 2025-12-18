from flask import Flask, render_template, request
import pickle
import re

app = Flask(__name__)

# Load trained model & vectorizer
with open("model/model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Text cleaning (same logic as training)
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    return text.strip()

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    confidence_level = None

    if request.method == "POST":
        message = request.form.get("message", "")
        cleaned = clean_text(message)

        if cleaned:
            vector = vectorizer.transform([cleaned])
            pred = model.predict(vector)[0]
            proba = model.predict_proba(vector)[0].max()

            confidence = round(proba * 100, 2)

            # Confidence level logic
            if confidence < 60:
                confidence_level = "low"
            elif confidence < 85:
                confidence_level = "medium"
            else:
                confidence_level = "high"

            prediction = "SPAM" if pred == 1 else "HAM"

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        confidence_level=confidence_level
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)