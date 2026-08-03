import json
from pathlib import Path

import torch


def evaluate_model(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    class_correct = [0] * 10
    class_total = [0] * 10

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            predictions = logits.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            for label, prediction in zip(labels, predictions):
                class_total[label.item()] += 1
                if label.item() == prediction.item():
                    class_correct[label.item()] += 1

    class_accuracy = {
        str(index): class_correct[index] / class_total[index]
        for index in range(10)
        if class_total[index] > 0
    }
    return {
        "accuracy": correct / total,
        "class_accuracy": class_accuracy,
        "total_samples": total,
    }


def save_metrics(metrics, output_path: Path):
    output_path.parent.mkdir(exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
