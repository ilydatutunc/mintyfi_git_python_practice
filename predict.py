import sys
import joblib

if len(sys.argv) != 2:
    print("\nInvalid usage!")
    print("Usage: python predict.py \"your message here\"")
    print("Example: python predict.py \"hi ilayda, how are you?\"\n")
    sys.exit(1)

message = sys.argv[1].strip()

if not message:
    print("\nMessage is empty. Please enter a valid message.")
    print("Usage: python predict.py \"your message here\"")
    print("Example: python predict.py \"hi ilayda, how are you?\"\n")
    sys.exit(1)



model = joblib.load("best_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

vectorized = vectorizer.transform([message])

prediction = model.predict(vectorized)[0]


import json
import os

def save_prediction_to_json(message, prediction, filename="pred_samples.json"):
    data = []
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []
    # Yeni tahmini ekle
    data.append({"message": message, "prediction": prediction})
    # Dosyaya tekrar yaz
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


print(f"Predicted label: {prediction}")

save_prediction_to_json(message, prediction)
print(f"Prediction saved to pred_samples.json\n")