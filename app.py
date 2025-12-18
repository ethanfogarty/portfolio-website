from flask import Flask, render_template, url_for, request, jsonify
import os, requests

app = Flask(__name__)

TFSERVING_URL = os.getenv("TFSERVING_URL", "http://localhost:8501/v1/models/sam:predict")

@app.post("/api/generate")
def generate():
    prompt = (request.json or {}).get("prompt", "")

    # This payload depends on your TF Serving signature.
    # Most TF Serving REST endpoints accept {"instances": ...}
    payload = {"instances": [{"prompt": prompt}]}  # <-- adjust to your model's input key

    r = requests.post(TFSERVING_URL, json=payload, timeout=60)
    r.raise_for_status()
    data = r.json()

    # Adjust based on your output format
    # e.g. {"predictions":[{"text":"..."}]}
    text = data["predictions"][0].get("text") if "predictions" in data else str(data)
    return jsonify({"text": text})

@app.route('/')
def index():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)