import csv
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class FeedbackDigitDataset(Dataset):
    def __init__(self, labels_path: Path):
        self.labels_path = labels_path
        self.rows = []
        if labels_path.exists():
            with labels_path.open("r", encoding="utf-8", newline="") as file:
                reader = csv.DictReader(file)
                self.rows = [
                    row
                    for row in reader
                    if row.get("true_label")
                    and row["true_label"].isdigit()
                    and 0 <= int(row["true_label"]) <= 9
                    and (row.get("image_name") or row.get("image_path"))
                ]
        self.transform = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((28, 28)),
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image_name = row.get("image_name")
        image_path = self.labels_path.parent / "images" / image_name if image_name else self.labels_path.parent.parent / row["image_path"]
        image = Image.open(image_path)
        label = int(row["true_label"])
        return self.transform(image), label


def ensure_feedback_store(images_dir: Path, labels_path: Path):
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    if not labels_path.exists():
        with labels_path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["image_name", "prediction", "true_label", "confidence", "timestamp"])
