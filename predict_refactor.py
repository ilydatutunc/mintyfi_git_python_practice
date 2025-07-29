import joblib
import json
import os

# Model ve vectorizer yüklemesi
model = joblib.load("best_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")


def predict_message(message: str) -> str:
    if not message.strip():
        raise ValueError("Empty message provided.")
    vectorized = vectorizer.transform([message])
    return model.predict(vectorized)[0]


def save_prediction_to_json(message: str, prediction: str, filename="pred_samples.json"):
    data = []
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []
    data.append({"message": message, "prediction": prediction})
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("\nInvalid usage!")
        print("Usage: python predict.py \"your message here\"")
        sys.exit(1)

    message = sys.argv[1].strip()
    if not message:
        print("\nMessage is empty.")
        sys.exit(1)

    prediction = predict_message(message)
    print(f"Predicted label: {prediction}")
    save_prediction_to_json(message, prediction)
    print("Prediction saved to pred_samples.json\n")
