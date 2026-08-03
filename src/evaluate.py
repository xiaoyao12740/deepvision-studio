import json
from pathlib import Path

import numpy as np
import torch


def evaluate_model(model, loader, device, loss_fn=None):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    class_correct = [0] * 10
    class_total = [0] * 10
    confusion_matrix = np.zeros((10, 10), dtype=int)

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            if loss_fn is not None:
                total_loss += loss_fn(logits, labels).item() * labels.size(0)
            predictions = logits.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            for label, prediction in zip(labels, predictions):
                label_index = label.item()
                prediction_index = prediction.item()
                class_total[label_index] += 1
                confusion_matrix[label_index, prediction_index] += 1
                if label_index == prediction_index:
                    class_correct[label_index] += 1

    class_accuracy = {
        str(index): class_correct[index] / class_total[index]
        for index in range(10)
        if class_total[index] > 0
    }
    per_class = {}
    for index in range(10):
        true_positive = int(confusion_matrix[index, index])
        false_positive = int(confusion_matrix[:, index].sum() - true_positive)
        false_negative = int(confusion_matrix[index, :].sum() - true_positive)
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        f1_score = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[str(index)] = {
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "support": int(class_total[index]),
        }

    macro_precision = sum(row["precision"] for row in per_class.values()) / len(per_class)
    macro_recall = sum(row["recall"] for row in per_class.values()) / len(per_class)
    macro_f1 = sum(row["f1_score"] for row in per_class.values()) / len(per_class)

    return {
        "accuracy": correct / total,
        "loss": total_loss / total if loss_fn is not None and total else None,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "class_accuracy": class_accuracy,
        "per_class": per_class,
        "confusion_matrix": confusion_matrix.tolist(),
        "total_samples": total,
    }


def save_metrics(metrics, output_path: Path):
    output_path.parent.mkdir(exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
