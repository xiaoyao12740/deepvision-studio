import json
from pathlib import Path
import random
import csv
import time

import numpy as np
import torch
from torch import nn

from dataset import build_mnist_loaders
from evaluate import evaluate_model, save_metrics
from models import build_model
from model_registry import next_version_path, write_latest
from reporter import build_report
from visualize import save_sample_predictions, save_training_curve


def load_config(project_root: Path):
    return json.loads((project_root / "config.json").read_text(encoding="utf-8"))


def resolve(project_root: Path, relative_path: str):
    return project_root / relative_path


def select_device(config):
    if config["training"]["device"] == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(config["training"]["device"])


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_one_epoch(model, loader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0.0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = loss_fn(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def save_train_log(history, output_path: Path):
    output_path.parent.mkdir(exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["epoch", "loss", "accuracy"])
        writer.writeheader()
        for row in history:
            writer.writerow(
                {
                    "epoch": row["epoch"],
                    "loss": f"{row['loss']:.6f}",
                    "accuracy": f"{row['accuracy']:.6f}",
                }
            )


def count_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def main():
    project_root = Path(__file__).resolve().parents[1]
    config = load_config(project_root)
    set_seed(config["training"]["random_state"])
    device = select_device(config)

    data_dir = resolve(project_root, config["data"]["data_dir"])
    train_loader, test_loader = build_mnist_loaders(
        data_dir,
        batch_size=config["data"]["batch_size"],
        test_batch_size=config["data"]["test_batch_size"],
    )

    model_type = config["training"]["model_type"]
    model = build_model(model_type).to(device)
    parameter_count = count_parameters(model)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])

    history = []
    started_at = time.perf_counter()
    for epoch in range(1, config["training"]["epochs"] + 1):
        loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        metrics = evaluate_model(model, test_loader, device)
        history.append({"epoch": epoch, "loss": loss, "accuracy": metrics["accuracy"]})
        print(f"Epoch {epoch}: loss={loss:.4f}, accuracy={metrics['accuracy']:.4f}")
    training_seconds = time.perf_counter() - started_at

    final_metrics = evaluate_model(model, test_loader, device)
    final_metrics["history"] = history
    final_metrics["model_type"] = model_type
    final_metrics["device"] = str(device)
    final_metrics["parameter_count"] = parameter_count
    final_metrics["training_seconds"] = round(training_seconds, 3)

    model_path, version = next_version_path(project_root, model_type)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_type": model_type,
            "metrics": final_metrics,
            "parameter_count": parameter_count,
        },
        model_path,
    )
    write_latest(project_root, model_path, version, model_type, final_metrics)

    save_metrics(final_metrics, resolve(project_root, config["outputs"]["metrics_path"]))
    save_train_log(history, resolve(project_root, config["outputs"]["train_log_path"]))
    save_training_curve(history, resolve(project_root, config["outputs"]["training_curve_path"]))
    save_sample_predictions(model, test_loader, device, resolve(project_root, config["outputs"]["sample_predictions_path"]))
    build_report(
        model_type=model_type,
        epochs=config["training"]["epochs"],
        metrics=final_metrics,
        report_path=resolve(project_root, config["outputs"]["report_path"]),
        include_training_curve=True,
    )

    print(f"Saved model version v{version} to: {model_path}")
    print(f"Saved metrics to: {resolve(project_root, config['outputs']['metrics_path'])}")
    print(f"Saved training log to: {resolve(project_root, config['outputs']['train_log_path'])}")
    print(f"Saved training curve to: {resolve(project_root, config['outputs']['training_curve_path'])}")
    print(f"Saved report to: {resolve(project_root, config['outputs']['report_path'])}")


if __name__ == "__main__":
    main()
