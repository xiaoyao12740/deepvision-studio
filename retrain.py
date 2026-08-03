import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader
from torchvision import datasets, transforms

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from evaluate import evaluate_model, save_metrics
from feedback_dataset import FeedbackDigitDataset, ensure_feedback_store
from inference import resolve, select_device
from models import build_model
from model_registry import next_version_path, write_latest
from reporter import build_report
from train import count_parameters, save_train_log
from visualize import save_confusion_matrix, save_sample_predictions, save_training_curve


def load_config():
    return json.loads((PROJECT_ROOT / "config.json").read_text(encoding="utf-8"))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_retrain_loaders(config):
    transform = transforms.ToTensor()
    data_dir = resolve(PROJECT_ROOT, config["data"]["data_dir"])
    mnist_train = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST(data_dir, train=False, download=True, transform=transform)

    images_dir = resolve(PROJECT_ROOT, config["feedback"]["images_dir"])
    labels_path = resolve(PROJECT_ROOT, config["feedback"]["labels_path"])
    ensure_feedback_store(images_dir, labels_path)
    feedback_data = FeedbackDigitDataset(labels_path)

    train_data = ConcatDataset([mnist_train, feedback_data]) if len(feedback_data) else mnist_train
    train_loader = DataLoader(train_data, batch_size=config["data"]["batch_size"], shuffle=True)
    test_loader = DataLoader(mnist_test, batch_size=config["data"]["test_batch_size"], shuffle=False)
    return train_loader, test_loader, len(feedback_data)


def train_one_epoch_with_progress(
    model,
    loader,
    loss_fn,
    optimizer,
    device,
    epoch,
    total_epochs,
    completed_steps,
    total_steps,
    last_percent,
):
    model.train()
    total_loss = 0.0
    for batch_index, (images, labels) in enumerate(loader, start=1):
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = loss_fn(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        current_step = completed_steps + batch_index
        percent = min(99, int(current_step * 100 / max(total_steps, 1)))
        if percent > last_percent:
            running_loss = total_loss / batch_index
            print(
                f"PROGRESS percent={percent} epoch={epoch} total_epochs={total_epochs} "
                f"batch={batch_index} total_batches={len(loader)} loss={running_loss:.4f}",
                flush=True,
            )
            last_percent = percent

    return total_loss / len(loader), completed_steps + len(loader), last_percent


def main():
    config = load_config()
    set_seed(config["training"]["random_state"])
    device = select_device(config)

    train_loader, test_loader, feedback_count = build_retrain_loaders(config)
    model_type = config["training"].get("model_type", "cnn_optimized")
    model = build_model(model_type).to(device)
    parameter_count = count_parameters(model)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])

    history = []
    started_at = time.perf_counter()
    total_epochs = config["training"]["epochs"]
    total_steps = len(train_loader) * total_epochs
    completed_steps = 0
    last_percent = -1
    for epoch in range(1, total_epochs + 1):
        loss, completed_steps, last_percent = train_one_epoch_with_progress(
            model,
            train_loader,
            loss_fn,
            optimizer,
            device,
            epoch,
            total_epochs,
            completed_steps,
            total_steps,
            last_percent,
        )
        metrics = evaluate_model(model, test_loader, device, loss_fn)
        history.append({"epoch": epoch, "loss": loss, "accuracy": metrics["accuracy"]})
        print(f"Retrain epoch {epoch}: loss={loss:.4f}, accuracy={metrics['accuracy']:.4f}", flush=True)
        print(
            f"PROGRESS percent={min(99, int(completed_steps * 100 / max(total_steps, 1)))} "
            f"epoch={epoch} total_epochs={total_epochs} "
            f"loss={loss:.4f} accuracy={metrics['accuracy']:.4f}",
            flush=True,
        )
    training_seconds = time.perf_counter() - started_at

    final_metrics = evaluate_model(model, test_loader, device, loss_fn)
    final_metrics["history"] = history
    final_metrics["model_type"] = model_type
    final_metrics["device"] = str(device)
    final_metrics["feedback_samples"] = feedback_count
    final_metrics["parameter_count"] = parameter_count
    final_metrics["training_seconds"] = round(training_seconds, 3)

    model_path, version = next_version_path(PROJECT_ROOT, model_type)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_type": model_type,
            "metrics": final_metrics,
            "feedback_samples": feedback_count,
            "parameter_count": parameter_count,
        },
        model_path,
    )
    latest_path = write_latest(PROJECT_ROOT, model_path, version, model_type, final_metrics)

    save_metrics(final_metrics, resolve(PROJECT_ROOT, config["outputs"]["metrics_path"]))
    save_train_log(history, resolve(PROJECT_ROOT, config["outputs"]["train_log_path"]))
    save_training_curve(history, resolve(PROJECT_ROOT, config["outputs"]["training_curve_path"]))
    save_confusion_matrix(final_metrics["confusion_matrix"], resolve(PROJECT_ROOT, config["outputs"]["confusion_matrix_path"]))
    save_sample_predictions(model, test_loader, device, resolve(PROJECT_ROOT, config["outputs"]["sample_predictions_path"]))
    build_report(
        model_type=model_type,
        epochs=config["training"]["epochs"],
        metrics=final_metrics,
        report_path=resolve(PROJECT_ROOT, config["outputs"]["report_path"]),
        include_training_curve=True,
    )

    print(f"Feedback samples used: {feedback_count}", flush=True)
    print(f"Published model version v{version}: {model_path}", flush=True)
    print(f"Updated latest pointer: {latest_path}", flush=True)
    print(
        f"RESULT accuracy={final_metrics['accuracy']:.4f} "
        f"feedback_samples={feedback_count} model_path={model_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
