from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


def save_sample_predictions(model, loader, device, output_path: Path):
    model.eval()
    images, labels = next(iter(loader))
    images = images.to(device)

    with torch.no_grad():
        predictions = model(images).argmax(dim=1).cpu()

    output_path.parent.mkdir(exist_ok=True)
    plt.figure(figsize=(10, 5))
    for index in range(10):
        plt.subplot(2, 5, index + 1)
        plt.imshow(images[index].cpu().squeeze(), cmap="gray")
        plt.title(f"pred={predictions[index]}, true={labels[index]}")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def save_probability_chart(probabilities, output_path: Path):
    output_path.parent.mkdir(exist_ok=True)
    plt.figure(figsize=(7, 4))
    plt.bar(range(10), probabilities, color="#2563eb")
    plt.xticks(range(10))
    plt.xlabel("Digit")
    plt.ylabel("Probability")
    plt.title("Prediction Probability")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def save_training_curve(history, output_path: Path):
    output_path.parent.mkdir(exist_ok=True)
    epochs = [row["epoch"] for row in history]
    losses = [row["loss"] for row in history]
    accuracies = [row["accuracy"] for row in history]

    plt.figure(figsize=(8, 4.8))
    ax_loss = plt.gca()
    ax_acc = ax_loss.twinx()

    ax_loss.plot(epochs, losses, marker="o", color="#dc2626", label="Loss")
    ax_acc.plot(epochs, accuracies, marker="s", color="#2563eb", label="Accuracy")

    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_acc.set_ylabel("Accuracy")
    ax_loss.set_xticks(epochs)
    ax_loss.grid(True, linestyle="--", alpha=0.35)

    lines = ax_loss.get_lines() + ax_acc.get_lines()
    labels = [line.get_label() for line in lines]
    ax_loss.legend(lines, labels, loc="center right")
    plt.title("Training Curve")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def save_confusion_matrix(matrix, output_path: Path, title="Test Confusion Matrix"):
    output_path.parent.mkdir(exist_ok=True)
    plt.figure(figsize=(7, 6))
    image = plt.imshow(matrix, cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.xticks(range(10))
    plt.yticks(range(10))
    for y in range(10):
        for x in range(10):
            value = matrix[y][x]
            if value:
                plt.text(x, y, str(value), ha="center", va="center", color="black", fontsize=8)
    plt.colorbar(image, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
