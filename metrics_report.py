import json
from sklearn.metrics import classification_report
import torch

def save_classification_report(model, test_loader, device, filename="mlp_metrics.json"):
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_labels.extend(labels.tolist())
            all_preds.extend(preds.cpu().tolist())

    report = classification_report(all_labels, all_preds, output_dict=True)

    with open(filename, 'w') as f:
        json.dump(report, f, indent=4)

    print(f"Classification report saved to {filename}")
