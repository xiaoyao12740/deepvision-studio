import argparse
from pathlib import Path

from torchvision import datasets, transforms

from inference import load_config, load_model, predict_tensor, preprocess_digit_image, resolve, select_device
from visualize import save_probability_chart
from PIL import Image


def load_sample_tensor(project_root: Path, index: int):
    data = datasets.MNIST(project_root / "data", train=False, download=True, transform=transforms.ToTensor())
    image, label = data[index]
    return image.unsqueeze(0), label


def main():
    parser = argparse.ArgumentParser(description="Predict handwritten digit")
    parser.add_argument("--image", default=None, help="Path to an image file")
    parser.add_argument("--sample-index", type=int, default=0, help="MNIST test sample index")
    parser.add_argument("--invert", action="store_true", help="Force image inversion before prediction")
    parser.add_argument("--no-auto-invert", action="store_true", help="Disable automatic white-background inversion")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    config = load_config(project_root)
    device = select_device(config)
    model, _ = load_model(project_root, config, device)

    true_label = None
    if args.image:
        invert = True if args.invert else False if args.no_auto_invert else None
        image_tensor = preprocess_digit_image(Image.open(Path(args.image)), invert=invert)
    else:
        image_tensor, true_label = load_sample_tensor(project_root, args.sample_index)

    prediction, confidence, probabilities = predict_tensor(model, image_tensor, device)

    save_probability_chart(probabilities.numpy(), resolve(project_root, config["outputs"]["probability_chart_path"]))
    print(f"Prediction: {prediction}")
    if true_label is not None:
        print(f"True label: {true_label}")
    print(f"Confidence: {confidence:.4f}")


if __name__ == "__main__":
    main()
