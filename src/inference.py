import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, ImageStat
import torch
from torchvision import transforms

from models import build_model
from model_registry import latest_model_path


def load_config(project_root: Path):
    return json.loads((project_root / "config.json").read_text(encoding="utf-8"))


def resolve(project_root: Path, relative_path: str):
    return project_root / relative_path


def select_device(config=None):
    if config and config.get("training", {}).get("device") not in (None, "auto"):
        return torch.device(config["training"]["device"])
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(project_root: Path, config, device, model_path: Path | None = None):
    checkpoint_path = model_path or latest_model_path(project_root, config)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model not found: {checkpoint_path}. Run training first.")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = build_model(checkpoint["model_type"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def should_invert_for_mnist(image: Image.Image):
    grayscale = image.convert("L")
    mean_pixel = ImageStat.Stat(grayscale).mean[0]
    return mean_pixel > 127


def crop_digit_content(image: Image.Image):
    grayscale = image.convert("L")
    array = np.asarray(grayscale)
    mask = array > 20
    if not mask.any():
        return grayscale

    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    top, bottom = rows[0], rows[-1]
    left, right = cols[0], cols[-1]
    width = right - left + 1
    height = bottom - top + 1
    pad = max(4, int(max(width, height) * 0.22))
    box = (
        max(0, left - pad),
        max(0, top - pad),
        min(grayscale.width, right + pad + 1),
        min(grayscale.height, bottom + pad + 1),
    )
    cropped = grayscale.crop(box)
    side = max(cropped.width, cropped.height)
    square = Image.new("L", (side, side), 0)
    offset = ((side - cropped.width) // 2, (side - cropped.height) // 2)
    square.paste(cropped, offset)
    return square


def fit_to_mnist_canvas(image: Image.Image):
    array = np.asarray(image)
    rows, cols = np.where(array > 20)
    if len(rows) == 0:
        return image.resize((28, 28), Image.Resampling.LANCZOS)

    width, height = image.size
    scale = 20 / max(width, height)
    resized = image.resize(
        (max(1, round(width * scale)), max(1, round(height * scale))),
        Image.Resampling.LANCZOS,
    )

    canvas = Image.new("L", (28, 28), 0)
    small = np.asarray(resized)
    rows, cols = np.where(small > 20)
    if len(rows) == 0:
        return canvas
    center_y = float((rows * small[rows, cols]).sum() / small[rows, cols].sum())
    center_x = float((cols * small[rows, cols]).sum() / small[rows, cols].sum())
    left = int(round(14 - center_x))
    top = int(round(14 - center_y))
    canvas.paste(resized, (left, top))
    return canvas


def preprocess_digit_image(image: Image.Image, invert: bool | None = None):
    image = image.convert("L")
    if invert is None:
        invert = should_invert_for_mnist(image)
    if invert:
        image = ImageOps.invert(image)
    image = fit_to_mnist_canvas(crop_digit_content(image))
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    return transform(image).unsqueeze(0)


def predict_tensor(model, image_tensor, device):
    with torch.no_grad():
        logits = model(image_tensor.to(device))
        probabilities = torch.softmax(logits, dim=1).cpu().squeeze()
        prediction = int(probabilities.argmax().item())
        confidence = float(probabilities[prediction].item())
    return prediction, confidence, probabilities
