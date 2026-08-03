# DeepVision Studio

DeepVision Studio is a small PyTorch product prototype for handwritten digit recognition. It connects model training, Streamlit inference, user correction, feedback analysis, offline retraining, model version management, and Docker deployment.

## What This Project Shows

- A working CNN digit recognizer trained on MNIST.
- A Streamlit drawing interface for `0-999` handwritten number recognition.
- Input preprocessing for inversion, cropping, resizing, and MNIST-style centering.
- User feedback collection with images, predictions, true labels, confidence, and timestamps.
- Feedback analysis that exposes common mistake patterns.
- Offline retraining instead of risky one-sample online updates.
- Model lifecycle management with versioned `.pt` files and `models/latest.json`.
- Docker deployment for reproducible local demos.

## Project Structure

```text
03-deepvision-studio/
  app.py
  retrain.py
  feedback_analysis.py
  config.json
  Dockerfile
  docker-compose.yml
  requirements.txt
  feedback/
    images/
    labels.csv
  models/
    digit_classifier.pt
    digit_classifier_v1.pt
    latest.json
  outputs/
    metrics.json
    feedback_analysis.json
    confusion_feedback.png
    train_log.csv
    training_curve.png
    report.html
  src/
    models.py
    inference.py
    model_registry.py
    train.py
    evaluate.py
    feedback_dataset.py
    visualize.py
```

## Run The App

From the `ML-DL-Projects` folder:

```powershell
.\.venv\Scripts\streamlit.exe run .\03-deepvision-studio\app.py --server.port 8503
```

Open:

```text
http://localhost:8503
```

The app loads the active model from `models/latest.json` when it exists. If no latest pointer exists yet, it falls back to `models/digit_classifier.pt`.

## Train

```powershell
.\.venv\Scripts\python.exe .\03-deepvision-studio\src\train.py
```

The default model type is `cnn_optimized`, which adds BatchNorm and Dropout on top of the earlier CNN baseline. You can still switch `training.model_type` in `config.json` to:

- `mlp`
- `cnn`
- `cnn_optimized`

Each training run now saves a new model version:

```text
models/digit_classifier_v1.pt
models/digit_classifier_v2.pt
models/latest.json
```

`latest.json` records the active version, model type, accuracy, feedback count, and creation time.

## Feedback Loop

```text
User drawing
  -> CNN prediction
  -> user correction
  -> feedback/images + feedback/labels.csv
  -> feedback analysis
  -> offline retraining
  -> new model version
  -> latest.json points to active model
```

The app no longer encourages immediate training from tiny feedback batches. The default suggested threshold is `100` single-digit feedback samples, controlled by:

```json
"min_feedback_before_update": 100
```

Manual offline updates are still available for demos and experiments.

## Analyze Feedback

Run:

```powershell
.\.venv\Scripts\python.exe .\03-deepvision-studio\feedback_analysis.py
```

Outputs:

```text
outputs/feedback_analysis.json
outputs/confusion_feedback.png
```

The analysis includes:

- Total feedback rows.
- Single-digit trainable feedback rows.
- Single-digit mistakes.
- Most common true-label-to-prediction error pairs.
- Most frequently mistaken true digits.
- Lowest-confidence feedback samples.

## Offline Retraining

```powershell
.\.venv\Scripts\python.exe .\03-deepvision-studio\retrain.py
```

Retraining combines MNIST training data with valid single-digit feedback samples, evaluates on MNIST test data, writes a new versioned checkpoint, and updates `models/latest.json`.

## Docker Deployment

Build and run with Docker Compose:

```powershell
docker compose up --build
```

Open:

```text
http://localhost:8503
```

The compose file mounts `feedback/`, `models/`, and `outputs/` so feedback data, model versions, and reports remain available after the container stops.

## Productization Roadmap

Completed in this stage:

- Canvas resize preservation and smoother UI behavior.
- MNIST-style input centering.
- Optimized CNN architecture.
- Feedback analysis script and confusion matrix export.
- Versioned model publishing with `latest.json`.
- Safer update threshold.
- Docker deployment files.

Optional future work:

- Add a small admin-only update screen.
- Compare `cnn` vs `cnn_optimized` in a dedicated experiment table.
- Add ResNet18 only if the project needs to demonstrate transfer learning.
