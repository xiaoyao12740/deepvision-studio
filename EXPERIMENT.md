# Experiment Record

## Purpose

The experiment compares a basic fully connected network and a convolutional neural network on MNIST handwritten digit recognition. The goal is to document why CNNs are better suited for image tasks and how the project evolves from model training into an AI application workflow.

本实验比较普通全连接网络和卷积神经网络在 MNIST 手写数字识别任务上的表现，并记录 CNN 为什么更适合图像任务，以及项目如何从模型训练扩展为 AI 应用闭环。

## Models

MLP:

```text
Flatten -> Linear -> ReLU -> Linear -> ReLU -> Linear
```

CNN:

```text
Conv2d -> ReLU -> MaxPool2d -> Conv2d -> ReLU -> MaxPool2d -> Linear
```

## Training Parameters

| Item | Value |
| --- | --- |
| Epochs | 3 |
| Learning rate | 0.001 |
| Batch size | 64 |
| Optimizer | Adam |
| Loss | CrossEntropyLoss |
| Dataset | MNIST |

## Result Summary

| Model | Accuracy | Parameters | Training Time |
| --- | ---: | ---: | ---: |
| MLP | 0.9768 | 235,146 | 23.830 s |
| CNN | 0.9879 | 206,922 | 45.532 s |

The comparison result is saved in:

```text
outputs/model_comparison.csv
```

The main training script records per-epoch values in:

```text
outputs/train_log.csv
```

It also exports:

```text
outputs/training_curve.png
```

## Why CNN Fits Image Tasks

- Local receptive fields allow convolution layers to focus on nearby pixels.
- Parameter sharing reduces the number of weights compared with fully connected image inputs.
- Pooling and convolution preserve useful spatial patterns.
- CNNs can learn strokes, edges, and local shapes before classification.

中文分析：

- 局部感受野让卷积层关注相邻像素区域
- 参数共享减少了直接全连接图像输入带来的参数规模
- 卷积和池化能保留有用的空间结构
- CNN 可以先学习笔画、边缘和局部形状，再进行分类

## Application Upgrade

The project now extends beyond offline MNIST training:

- `app.py` provides Streamlit-based drawing and prediction.
- `feedback/images` stores user-submitted digit images.
- `feedback/labels.csv` stores prediction, true label, confidence, and timestamp.
- `retrain.py` combines MNIST with feedback data and outputs `digit_classifier.pt`.

中文说明：

- `app.py` 提供 Streamlit 手写绘制和识别页面
- `feedback/images` 保存用户提交的数字图片
- `feedback/labels.csv` 保存预测结果、真实标签、置信度和时间戳
- `retrain.py` 合并 MNIST 与反馈数据，输出 `digit_classifier.pt`

## Training Strategy

The app does not perform online training. User corrections are collected as data first. Model updates happen later through offline retraining and overwriting the active model.

应用不做在线训练。用户纠错先进入反馈数据池，后续通过离线重新训练和模型版本发布完成迭代。
