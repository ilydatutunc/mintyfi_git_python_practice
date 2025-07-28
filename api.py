from flask import Flask, request, jsonify
import joblib
import numpy as np


app = Flask(__name__)

model = joblib.load("best_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' field in request"}), 400

    message = data["text"].strip()

    if not message:
        return jsonify({"error": "Empty message"}), 400


    vectorized = vectorizer.transform([message])

    prediction = model.predict(vectorized)[0]
    probability = float(np.max(model.predict_proba(vectorized)[0]))

   
    return jsonify({
        "prediction": prediction,
        "probability": probability
    })

if __name__ == "__main__":
    app.run(debug=True)
