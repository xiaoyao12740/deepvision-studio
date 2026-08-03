import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent


def is_single_digit(value: str):
    return value.isdigit() and 0 <= int(value) <= 9


def load_feedback(labels_path: Path):
    if not labels_path.exists():
        return []
    with labels_path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def build_confusion(rows):
    matrix = np.zeros((10, 10), dtype=int)
    for row in rows:
        prediction = row.get("prediction", "").strip()
        true_label = row.get("true_label", "").strip()
        if is_single_digit(prediction) and is_single_digit(true_label):
            matrix[int(true_label), int(prediction)] += 1
    return matrix


def save_confusion_plot(matrix, output_path: Path):
    output_path.parent.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_title("Feedback Confusion Matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    for y in range(10):
        for x in range(10):
            value = matrix[y, x]
            if value:
                ax.text(x, y, str(value), ha="center", va="center", color="black")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def analyze(rows):
    total = len(rows)
    single_digit_rows = [
        row for row in rows
        if is_single_digit(row.get("prediction", "")) and is_single_digit(row.get("true_label", ""))
    ]
    mistakes = [
        row for row in single_digit_rows
        if row["prediction"].strip() != row["true_label"].strip()
    ]
    error_pairs = Counter((row["true_label"].strip(), row["prediction"].strip()) for row in mistakes)
    true_label_counts = Counter(row["true_label"].strip() for row in mistakes)
    low_confidence = sorted(
        rows,
        key=lambda row: float(row.get("confidence") or 0),
    )[:10]
    return {
        "total_feedback": total,
        "single_digit_feedback": len(single_digit_rows),
        "single_digit_mistakes": len(mistakes),
        "top_error_pairs": [
            {"true_label": true_label, "prediction": prediction, "count": count}
            for (true_label, prediction), count in error_pairs.most_common(10)
        ],
        "most_mistaken_true_labels": [
            {"true_label": label, "count": count}
            for label, count in true_label_counts.most_common(10)
        ],
        "lowest_confidence_samples": [
            {
                "image_name": row.get("image_name"),
                "prediction": row.get("prediction"),
                "true_label": row.get("true_label"),
                "confidence": float(row.get("confidence") or 0),
                "timestamp": row.get("timestamp"),
            }
            for row in low_confidence
        ],
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze DeepVision Studio feedback data.")
    parser.add_argument("--labels", default="feedback/labels.csv")
    parser.add_argument("--output-json", default="outputs/feedback_analysis.json")
    parser.add_argument("--confusion-png", default="outputs/confusion_feedback.png")
    args = parser.parse_args()

    labels_path = PROJECT_ROOT / args.labels
    rows = load_feedback(labels_path)
    summary = analyze(rows)
    matrix = build_confusion(rows)

    output_json = PROJECT_ROOT / args.output_json
    output_json.parent.mkdir(exist_ok=True)
    output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    save_confusion_plot(matrix, PROJECT_ROOT / args.confusion_png)

    print(f"Feedback rows: {summary['total_feedback']}")
    print(f"Single-digit rows: {summary['single_digit_feedback']}")
    print(f"Single-digit mistakes: {summary['single_digit_mistakes']}")
    print(f"Saved analysis: {output_json}")
    print(f"Saved confusion matrix: {PROJECT_ROOT / args.confusion_png}")


if __name__ == "__main__":
    main()
