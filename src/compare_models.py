import csv
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn

from dataset import build_mnist_loaders
from evaluate import evaluate_model
from inference import resolve, select_device
from models import build_model
from train import count_parameters, train_one_epoch


def load_config(project_root: Path):
    return json.loads((project_root / "config.json").read_text(encoding="utf-8"))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_model(model_type, train_loader, test_loader, config, device):
    set_seed(config["training"]["random_state"])
    model = build_model(model_type).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])

    started_at = time.perf_counter()
    final_loss = 0.0
    for _ in range(config["training"]["epochs"]):
        final_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
    training_seconds = time.perf_counter() - started_at
    metrics = evaluate_model(model, test_loader, device)
    return {
        "model": model_type,
        "accuracy": metrics["accuracy"],
        "parameters": count_parameters(model),
        "training_seconds": round(training_seconds, 3),
        "final_loss": final_loss,
    }


def main():
    project_root = Path(__file__).resolve().parents[1]
    config = load_config(project_root)
    device = select_device(config)
    train_loader, test_loader = build_mnist_loaders(
        resolve(project_root, config["data"]["data_dir"]),
        batch_size=config["data"]["batch_size"],
        test_batch_size=config["data"]["test_batch_size"],
    )

    rows = [run_model(model_type, train_loader, test_loader, config, device) for model_type in ["mlp", "cnn"]]
    output_path = resolve(project_root, config["outputs"]["model_comparison_path"])
    output_path.parent.mkdir(exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["model", "accuracy", "parameters", "training_seconds", "final_loss"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    for row in rows:
        print(
            f"{row['model']}: accuracy={row['accuracy']:.4f}, "
            f"parameters={row['parameters']}, training_seconds={row['training_seconds']}"
        )
    print(f"Saved model comparison to: {output_path}")


if __name__ == "__main__":
    main()
