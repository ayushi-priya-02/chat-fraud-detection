from flask import Flask, render_template, request
import pickle
import os
import re

app = Flask(__name__)

# -------------------------------
# Load trained ML model & vectorizer
# -------------------------------
MODEL_PATH = os.path.join("model", "model.pkl")
VECTORIZER_PATH = os.path.join("model", "vectorizer.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(VECTORIZER_PATH, "rb") as f:
    vectorizer = pickle.load(f)

# -------------------------------
# Phone number extraction
# -------------------------------
def extract_phone_numbers(text):
    pattern = r"\b\d{10}\b"
    return re.findall(pattern, text)

# -------------------------------
# Phone number analysis
# -------------------------------
def analyze_phone_number(number):
    if len(number) != 10:
        return "❌ Invalid"

    if len(set(number)) == 1:
        return "⚠️ Suspicious (all digits same)"

    if number[0] not in ["6", "7", "8", "9"]:
        return "⚠️ Suspicious (invalid start)"

    if number[:5] == number[5:]:
        return "⚠️ Suspicious (repeated pattern)"

    return "✅ Looks Valid"

# -------------------------------
# URL extraction
# -------------------------------
def extract_urls(text):
    pattern = r"(https?://[^\s]+|www\.[^\s]+)"
    return re.findall(pattern, text)

# -------------------------------
# URL phishing analysis
# -------------------------------
def analyze_url(url):
    suspicious_keywords = [
        "login", "verify", "update", "secure",
        "bank", "free", "offer", "reward"
    ]

    if any(word in url.lower() for word in suspicious_keywords):
        return "⚠️ Suspicious URL"

    if url.count("-") > 3:
        return "⚠️ Suspicious URL"

    if len(url) > 60:
        return "⚠️ Suspicious URL"

    return "✅ Looks Safe"

# -------------------------------
# Main route
# -------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    phone_analysis = []
    url_analysis = []

    if request.method == "POST":
        message = request.form.get("message")

        if message:
            # -------- ML Spam / Ham Detection --------
            text_vector = vectorizer.transform([message])
            pred = model.predict(text_vector)[0]
            prob = model.predict_proba(text_vector)[0]

            if pred == 1:
                prediction = "SPAM 🚨"
                confidence = round(prob[1] * 100, 2)
            else:
                prediction = "HAM ✅"
                confidence = round(prob[0] * 100, 2)

            # -------- Phone Number Analysis --------
            phones = extract_phone_numbers(message)
            for phone in phones:
                phone_analysis.append((phone, analyze_phone_number(phone)))

            # -------- URL Analysis --------
            urls = extract_urls(message)
            for url in urls:
                url_analysis.append((url, analyze_url(url)))

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        phone_analysis=phone_analysis,
        url_analysis=url_analysis
    )

# -------------------------------
# Run app (Local + Render)
# -------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
