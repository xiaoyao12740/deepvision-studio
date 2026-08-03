import json
from datetime import datetime
from pathlib import Path


def latest_model_path(project_root: Path, config):
    latest_path = project_root / "models" / "latest.json"
    if latest_path.exists():
        latest = json.loads(latest_path.read_text(encoding="utf-8"))
        model_path = project_root / latest["model_path"]
        if model_path.exists():
            return model_path
    return project_root / config["outputs"]["model_path"]


def next_version_path(project_root: Path, model_type: str):
    models_dir = project_root / "models"
    models_dir.mkdir(exist_ok=True)
    versions = []
    for path in models_dir.glob("digit_classifier_v*.pt"):
        stem = path.stem.replace("digit_classifier_v", "")
        if stem.isdigit():
            versions.append(int(stem))
    version = max(versions, default=0) + 1
    return models_dir / f"digit_classifier_v{version}.pt", version


def write_latest(project_root: Path, model_path: Path, version: int, model_type: str, metrics: dict):
    latest = {
        "version": version,
        "model_type": model_type,
        "model_path": model_path.relative_to(project_root).as_posix(),
        "accuracy": metrics.get("accuracy"),
        "feedback_samples": metrics.get("feedback_samples", 0),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    latest_path = project_root / "models" / "latest.json"
    latest_path.write_text(json.dumps(latest, ensure_ascii=False, indent=2), encoding="utf-8")
    return latest_path
