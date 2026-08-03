# DeepVision Studio

DeepVision Studio is a lightweight deep learning application for handwritten digit recognition. It turns the MNIST dataset into a complete local workflow: CNN training, Streamlit drawing-based inference, user correction, feedback collection, feedback analysis, offline retraining, model version management, reporting, and Docker deployment.

DeepVision Studio 是一个面向手写数字识别的轻量级深度学习应用。项目基于 MNIST 数据集，完整串联了 CNN 模型训练、Streamlit 手写绘制推理、用户纠错、反馈收集、反馈分析、离线再训练、模型版本管理、结果报告和 Docker 本地部署。

## Positioning

This project is designed as a small product-style computer vision demo rather than only a training script. It shows how an image classification model can be wrapped into an interactive application where users can draw digits, inspect prediction confidence, correct mistakes, and feed those corrections back into an offline improvement workflow.

本项目定位为一个产品化的小型计算机视觉演示，而不是单独的训练脚本。它展示了如何把图像分类模型封装成可交互应用：用户可以手写数字、查看预测置信度、纠正错误，并把这些纠错样本纳入后续离线优化流程。

The current model focuses on MNIST-style handwritten digits. The app includes an experimental multi-digit mode for `0-999` by character segmentation plus per-digit CNN classification, but the core training data and evaluation target remain single MNIST digits.

当前模型主要面向 MNIST 风格的单个手写数字。应用中包含实验性的 `0-999` 多位数字模式，其实现方式是字符分割加逐位 CNN 分类，但核心训练数据和评估目标仍然是 MNIST 单数字识别。

## Highlights

- CNN-based handwritten digit recognition
- Streamlit canvas interface for drawing digits
- MNIST-style preprocessing with inversion, cropping, resizing, and centering
- Single-digit prediction and experimental segmentation-based multi-digit workflow
- Prediction confidence display and probability visualization
- User feedback collection with image snapshots, predictions, labels, confidence, and timestamps
- Feedback analysis with common mistake patterns and confusion matrix export
- Offline retraining from MNIST plus valid feedback samples
- Versioned model registry with `models/latest.json`
- HTML report and training visualizations
- Docker Compose configuration for local deployment

中文概览：

- 使用 CNN 完成手写数字识别
- 提供 Streamlit 画布交互界面
- 包含反色、裁剪、缩放、居中等 MNIST 风格预处理
- 支持单数字识别和实验性的字符分割式多位数字流程
- 展示预测置信度和概率分布图
- 收集用户反馈，包括图片、预测值、真实标签、置信度和时间戳
- 分析反馈数据，输出常见错误模式和混淆矩阵
- 使用 MNIST 与有效反馈样本进行离线再训练
- 使用 `models/latest.json` 管理当前模型版本
- 生成 HTML 报告和训练过程可视化结果
- 提供 Docker Compose 本地部署配置

## Core Workflow

```text
MNIST data
  -> CNN training
  -> model evaluation
  -> Streamlit drawing interface
  -> prediction and confidence display
  -> user correction feedback
  -> validation and cleaning
  -> feedback analysis
  -> offline retraining
  -> versioned model update
```

中文流程：

```text
MNIST 数据
  -> CNN 模型训练
  -> 模型评估
  -> Streamlit 手写交互界面
  -> 预测与置信度展示
  -> 用户纠错反馈
  -> 样本校验与清洗
  -> 反馈数据分析
  -> 离线再训练
  -> 模型版本更新
```

## Outputs

- `outputs/metrics.json`
- `outputs/train_log.csv`
- `outputs/training_curve.png`
- `outputs/confusion_matrix.png`
- `outputs/sample_predictions.png`
- `outputs/probability_chart.png`
- `outputs/feedback_analysis.json`
- `outputs/confusion_feedback.png`
- `outputs/model_comparison.csv`
- `outputs/report.html`
- `models/digit_classifier.pt`
- `models/digit_classifier_v*.pt`
- `models/latest.json`

These outputs cover accuracy, loss, macro precision, macro recall, macro F1, per-class metrics, test confusion matrix, training logs, visual evaluation samples, prediction probability charts, feedback analysis results, model comparison records, HTML reporting, saved checkpoints, and the active model pointer.

这些输出覆盖了准确率、损失值、宏平均 precision、宏平均 recall、宏平均 F1、逐类别指标、测试集混淆矩阵、训练日志、样例预测图、预测概率图、反馈分析结果、模型对比记录、HTML 报告、已保存模型权重和当前模型版本指针，便于复现实验和检查项目完整性。

## Results

Latest local training run with `cnn_optimized` on MNIST test data:

| Metric | Value |
| --- | ---: |
| Accuracy | 0.9914 |
| Loss | 0.0247 |
| Macro Precision | 0.9913 |
| Macro Recall | 0.9914 |
| Macro F1 | 0.9914 |
| Test Samples | 10,000 |
| Trainable Parameters | 468,202 |
| Training Time | 289.5 seconds on CPU |

The weakest class-level recall in this run is digit `9` at `0.9802`, which makes the confusion matrix useful for checking whether later feedback reduces those mistakes.

当前本地训练结果使用 `cnn_optimized` 模型，并在 MNIST 测试集上评估：

| 指标 | 数值 |
| --- | ---: |
| 准确率 | 0.9914 |
| 损失值 | 0.0247 |
| 宏平均 Precision | 0.9913 |
| 宏平均 Recall | 0.9914 |
| 宏平均 F1 | 0.9914 |
| 测试样本数 | 10,000 |
| 可训练参数量 | 468,202 |
| 训练耗时 | CPU 上约 289.5 秒 |

本轮结果中召回率相对最低的是数字 `9`，为 `0.9802`。后续可以用测试集混淆矩阵和反馈混淆矩阵观察这类错误是否被反馈数据改善。

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
    confusion_matrix.png
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
    compare_models.py
    reporter.py
    visualize.py
```

## Run Locally

Install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Train the model:

```powershell
.\.venv\Scripts\python.exe .\src\train.py
```

Start the Streamlit app:

```powershell
.\.venv\Scripts\streamlit.exe run .\app.py --server.port 8503
```

Open:

```text
http://localhost:8503
```

中文说明：

- 先创建虚拟环境并安装依赖
- 运行训练脚本生成模型、指标和可视化结果
- 再启动 Streamlit 页面进行手写数字识别演示
- 本项目建议使用 `8503` 端口，便于和前两个本地机器学习演示项目并行运行

The app loads the active model from `models/latest.json` when it exists. If no active version pointer is available, it falls back to `models/digit_classifier.pt`.

应用会优先从 `models/latest.json` 读取当前启用的模型版本。如果该版本指针不存在，则回退加载 `models/digit_classifier.pt`。

## Model Options

- `mlp`
- `cnn`
- `cnn_optimized`

The default model type is `cnn_optimized`, which adds BatchNorm and Dropout on top of the earlier CNN baseline. The model type can be changed through `training.model_type` in `config.json`.

默认模型类型是 `cnn_optimized`，在基础 CNN 结构上加入 BatchNorm 和 Dropout。可以通过 `config.json` 中的 `training.model_type` 切换模型类型。

## Feedback Loop

```text
User drawing
  -> model prediction
  -> user correction
  -> feedback/images + feedback/labels.csv
  -> validation and cleaning
  -> feedback analysis
  -> offline retraining
  -> new model version
  -> latest.json points to active model
```

中文反馈闭环：

```text
用户手写输入
  -> 模型预测
  -> 用户纠错
  -> 保存反馈图片与标签记录
  -> 样本校验与清洗
  -> 分析反馈数据
  -> 离线再训练
  -> 生成新模型版本
  -> latest.json 指向当前启用模型
```

The project intentionally avoids immediate one-sample online updates. A more production-like workflow would validate feedback, clean invalid samples, optionally review labels manually, and then expand the training set. The default suggested threshold is `100` valid single-digit feedback samples, controlled by:

项目刻意避免基于单个样本立即在线更新模型。更接近真实业务的流程应先校验反馈、清洗无效样本、必要时进行人工复核，然后再扩充训练集。默认建议至少收集 `100` 条有效单数字反馈样本后再更新模型，该阈值由以下配置控制：

```json
"min_feedback_before_update": 100
```

Manual offline retraining is still available for demos and experiments:

仍然可以手动运行离线再训练，用于演示和实验：

```powershell
.\.venv\Scripts\python.exe .\retrain.py
```

## Analyze Feedback

```powershell
.\.venv\Scripts\python.exe .\feedback_analysis.py
```

Generated files:

```text
outputs/feedback_analysis.json
outputs/confusion_feedback.png
```

The analysis includes total feedback rows, trainable single-digit rows, single-digit mistakes, common true-label-to-prediction error pairs, frequently mistaken true digits, and lowest-confidence samples.

反馈分析包含反馈总量、可用于训练的单数字反馈量、单数字错误样本、常见真实标签到预测标签的错误组合、最容易出错的真实数字，以及低置信度样本。

## Docker

```powershell
docker compose up --build
```

Open:

```text
http://localhost:8503
```

The compose file mounts `feedback/`, `models/`, and `outputs/` so feedback data, model versions, and reports remain available after the container stops.

`docker-compose.yml` 会挂载 `feedback/`、`models/` 和 `outputs/`，因此容器停止后，反馈数据、模型版本和报告文件仍会保留在本地。

## Known Limits

- The training target is MNIST-style single-digit classification.
- Multi-digit recognition is an experimental character-segmentation-and-classification workflow, not OCR, sequence recognition, or digit detection.
- Feedback images and model weights are local artifacts and are not committed to the repository by default.
- The app is local-first and single-user; it does not include authentication, database storage, or production monitoring.
- Retraining quality depends on the amount and correctness of collected feedback samples.

中文补充：

- 当前训练目标是 MNIST 风格的单数字分类
- 多位数字识别属于实验性的字符分割加逐位分类流程，不是 OCR、序列识别或数字检测系统
- 反馈图片和模型权重属于本地产物，默认不提交到仓库
- 当前应用以本地单用户演示为主，暂未加入登录鉴权、数据库和生产级监控
- 再训练效果取决于反馈样本的数量和标注准确性
