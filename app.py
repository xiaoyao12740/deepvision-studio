import csv
import json
import os
from datetime import datetime
from pathlib import Path
import re
import subprocess
import sys
import time
import uuid

import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from feedback_dataset import ensure_feedback_store
from inference import load_config, load_model, predict_tensor, preprocess_digit_image, resolve, select_device


def get_canvas_image(image_data):
    if image_data is None:
        return None
    array = np.asarray(image_data).astype("uint8")
    if array[:, :, 3].max() == 0:
        return None
    if array[:, :, :3].max() < 8:
        return None
    return Image.fromarray(array, mode="RGBA").convert("RGB")


def split_digit_regions(image: Image.Image):
    gray = np.asarray(image.convert("L"))
    mask = gray > 20
    if not mask.any():
        return []

    active_cols = np.where(mask.any(axis=0))[0]
    groups = []
    start = active_cols[0]
    previous = active_cols[0]
    min_gap = max(6, image.width // 45)
    for col in active_cols[1:]:
        if col - previous > min_gap:
            groups.append((start, previous))
            start = col
        previous = col
    groups.append((start, previous))

    regions = []
    for left, right in groups:
        region_mask = mask[:, left : right + 1]
        rows = np.where(region_mask.any(axis=1))[0]
        if len(rows) == 0:
            continue
        top = rows[0]
        bottom = rows[-1]
        pad = max(8, int(max(right - left + 1, bottom - top + 1) * 0.22))
        crop_box = (
            max(0, left - pad),
            max(0, top - pad),
            min(image.width, right + pad + 1),
            min(image.height, bottom + pad + 1),
        )
        crop = image.crop(crop_box)
        if crop.width >= 8 and crop.height >= 8:
            regions.append(crop)
    return regions


def predict_canvas_number(model, image: Image.Image, device):
    regions = split_digit_regions(image)
    if not regions:
        return None

    digits = []
    rows = []
    for index, region in enumerate(regions):
        image_tensor = preprocess_digit_image(region, invert=False)
        digit, confidence, probabilities = predict_tensor(model, image_tensor, device)
        digits.append(str(digit))
        rows.append(
            {
                "position": index + 1,
                "digit": digit,
                "confidence": confidence,
                "probabilities": probabilities.numpy(),
            }
        )

    value_text = "".join(digits)
    value = int(value_text)
    average_confidence = float(np.mean([row["confidence"] for row in rows]))
    return value, average_confidence, rows, regions


def append_feedback(labels_path: Path, image_name: str, prediction: str, true_label: str, confidence: float):
    with labels_path.open("a", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                image_name,
                prediction,
                true_label,
                f"{confidence:.6f}",
                datetime.now().isoformat(timespec="seconds"),
            ]
        )


def count_feedback_rows(labels_path: Path):
    if not labels_path.exists():
        return 0, 0
    with labels_path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    total = len(rows)
    trainable = sum(
        1
        for row in rows
        if row.get("true_label", "").isdigit() and 0 <= int(row["true_label"]) <= 9
    )
    return total, trainable


def normalize_actual_label(value):
    text = str(value).strip()
    chinese_digits = {
        "零": "0",
        "〇": "0",
        "一": "1",
        "二": "2",
        "两": "2",
        "三": "3",
        "四": "4",
        "五": "5",
        "六": "6",
        "七": "7",
        "八": "8",
        "九": "9",
        "洞": "0",
        "冻": "0",
        "幺": "1",
        "腰": "1",
        "要": "1",
        "贰": "2",
        "俩": "2",
        "仨": "3",
        "叁": "3",
        "肆": "4",
        "伍": "5",
        "陆": "6",
        "溜": "6",
        "柒": "7",
        "拐": "7",
        "捌": "8",
        "玖": "9",
        "久": "9",
        "酒": "9",
        "就": "9",
    }
    if text in chinese_digits:
        text = chinese_digits[text]
    else:
        converted = "".join(chinese_digits.get(character, character) for character in text)
        text = converted
    if not text.isdigit():
        raise ValueError
    number = int(text)
    if not 0 <= number <= 999:
        raise ValueError
    return str(number)


def set_actual_label(label_key: str, value: int):
    st.session_state[label_key] = str(value)


def format_duration(seconds: float):
    seconds = max(0, int(seconds))
    minutes, seconds = divmod(seconds, 60)
    if minutes:
        return f"{minutes}分{seconds}秒"
    return f"{seconds}秒"


def parse_progress_line(line: str):
    if not line.startswith("PROGRESS "):
        return None
    values = {}
    for item in line.split()[1:]:
        if "=" in item:
            key, value = item.split("=", 1)
            values[key] = value
    if "percent" not in values:
        return None
    return {
        "percent": int(values["percent"]),
        "epoch": int(values.get("epoch", 0)),
        "total_epochs": int(values.get("total_epochs", values.get("total", 0))),
        "batch": int(values.get("batch", 0)),
        "total_batches": int(values.get("total_batches", 0)),
        "loss": float(values.get("loss", 0.0)),
        "accuracy": float(values["accuracy"]) if "accuracy" in values else None,
    }


def read_latest_metrics(metrics_path: Path):
    if not metrics_path.exists():
        return None
    try:
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def show_update_judgement(metrics_path: Path, feedback_count: int):
    metrics = read_latest_metrics(metrics_path)
    if not metrics:
        st.warning("模型已更新，但暂时没有读到评估指标。Model updated, but metrics were not available.")
        return

    accuracy = float(metrics.get("accuracy", 0.0))
    threshold = 0.98
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("测试准确率 | Test Accuracy", f"{accuracy * 100:.2f}%")
    col_b.metric("反馈样本 | Feedback Used", str(metrics.get("feedback_samples", feedback_count)))
    col_c.metric("训练耗时 | Training Time", format_duration(metrics.get("training_seconds", 0)))

    if accuracy >= threshold:
        st.success("更新完成：测试准确率达到当前展示阈值 98%，旧模型已被覆盖。")
    else:
        st.warning("更新完成：旧模型已被覆盖，但测试准确率低于当前展示阈值 98%，建议继续积累更清晰的反馈样本。")


def run_retrain(progress_bar, status_box, log_box):
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    process = subprocess.Popen(
        [sys.executable, str(PROJECT_ROOT / "retrain.py")],
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )

    output_lines = []
    started_at = time.perf_counter()
    progress_bar.progress(0, text="准备更新模型 | Preparing model update")
    status_box.info("正在读取 MNIST 与反馈数据，稍后开始训练。")

    assert process.stdout is not None
    for raw_line in process.stdout:
        line = raw_line.strip()
        if not line:
            continue
        output_lines.append(line)
        progress = parse_progress_line(line)
        if progress:
            fraction = min(0.99, max(0.0, progress["percent"] / 100))
            elapsed = time.perf_counter() - started_at
            remaining = (elapsed / fraction - elapsed) if fraction > 0 else 0
            batch_text = ""
            if progress["batch"] and progress["total_batches"]:
                batch_text = f" | Batch {progress['batch']}/{progress['total_batches']}"
            progress_bar.progress(
                fraction,
                text=(
                    f"更新进度 {progress['percent']}% | "
                    f"Epoch {progress['epoch']}/{progress['total_epochs']}"
                    f"{batch_text} | 预计剩余 {format_duration(remaining)}"
                ),
            )
            if progress["accuracy"] is None:
                status_box.info(f"正在训练，当前 loss={progress['loss']:.4f}。")
            else:
                status_box.info(f"当前 loss={progress['loss']:.4f}，accuracy={progress['accuracy'] * 100:.2f}%。")
        log_box.code("\n".join(output_lines[-10:]))

    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError("\n".join(output_lines[-20:]) or "Retraining failed.")

    progress_bar.progress(1.0, text="更新完成 100% | Model update complete")
    status_box.success("模型文件已覆盖，下一次识别会加载更新后的模型。")
    load_digit_model.clear()
    return "\n".join(output_lines)


def run_model_update_ui(metrics_path: Path, feedback_count: int):
    progress_bar = st.progress(0, text="准备更新模型 | Preparing model update")
    status_box = st.empty()
    log_box = st.empty()
    output = run_retrain(progress_bar, status_box, log_box)
    show_update_judgement(metrics_path, feedback_count)
    return output


def clear_canvas_state():
    st.session_state["canvas_version"] = st.session_state.get("canvas_version", 0) + 1
    st.session_state.pop("canvas_snapshot", None)
    st.session_state.pop("canvas_snapshot_size", None)
    for key in [
        "last_prediction",
        "last_confidence",
        "last_digit_rows",
        "last_image",
        "last_regions",
    ]:
        st.session_state.pop(key, None)


def resize_canvas_snapshot(target_size: int):
    snapshot = st.session_state.get("canvas_snapshot")
    snapshot_size = st.session_state.get("canvas_snapshot_size")
    if snapshot is None or snapshot_size == target_size:
        return None
    return snapshot.resize((target_size, target_size), Image.Resampling.LANCZOS)


@st.cache_resource
def load_digit_model():
    config = load_config(PROJECT_ROOT)
    device = select_device(config)
    model, checkpoint = load_model(PROJECT_ROOT, config, device)
    return config, device, model, checkpoint


st.set_page_config(page_title="DeepVision Studio", layout="wide")
st.markdown(
    """
    <style>
    .block-container { padding-top: 2rem; }
    div[data-testid="stMetricValue"] { font-size: 2.3rem; }
    .dv-muted { color: #667085; font-size: 0.95rem; }
    iframe[title="streamlit_drawable_canvas.st_canvas"] {
      transition: width 260ms ease, height 260ms ease, transform 260ms ease;
      transform-origin: top left;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if "canvas_version" not in st.session_state:
    st.session_state["canvas_version"] = 0

st.title("DeepVision Studio")
st.caption("基于 PyTorch 的手写数字智能识别与用户反馈优化系统 | PyTorch digit recognition with a user feedback loop")

config, device, model, checkpoint = load_digit_model()
images_dir = resolve(PROJECT_ROOT, config["feedback"]["images_dir"])
labels_path = resolve(PROJECT_ROOT, config["feedback"]["labels_path"])
metrics_path = resolve(PROJECT_ROOT, config["outputs"]["metrics_path"])
ensure_feedback_store(images_dir, labels_path)

st.info("在下方黑色画布中写 0-999 的数字，点击识别；如果结果不对，在反馈区提交真实标签。反馈只会保存为数据，不会在线训练模型。")

left, right = st.columns([1.05, 1], gap="large")

with left:
    st.subheader("绘制数字 | Draw a Number")
    st.markdown(
        '<div class="dv-muted">支持 0-999。多位数会先分割成单个数字，再由 CNN 逐位识别。画布变大或变小时会保留当前字迹，橡皮擦模式可以局部清除字迹。</div>',
        unsafe_allow_html=True,
    )
    control_left, control_right = st.columns(2)
    with control_left:
        canvas_size = st.slider("画布大小 | Canvas size", min_value=200, max_value=600, value=360, step=20)
    with control_right:
        stroke_width = st.slider("笔刷粗细 | Brush width", min_value=8, max_value=36, value=18, step=2)

    mode_left, mode_right = st.columns([1, 1])
    with mode_left:
        erase_mode = st.toggle("橡皮擦 / Eraser", value=False)
    with mode_right:
        clear_clicked = st.button("重置画布 / Reset Canvas", type="secondary", use_container_width=True)
    if clear_clicked:
        clear_canvas_state()

    background_image = resize_canvas_snapshot(canvas_size)
    canvas_result = st_canvas(
        fill_color="rgba(0, 0, 0, 0)",
        stroke_width=stroke_width,
        stroke_color="#000000" if erase_mode else "#ffffff",
        background_color="#000000",
        background_image=background_image,
        width=canvas_size,
        height=canvas_size,
        drawing_mode="freedraw",
        display_toolbar=False,
        key=f"digit_canvas_{st.session_state['canvas_version']}_{canvas_size}",
    )

    canvas_image = get_canvas_image(canvas_result.image_data)
    if canvas_image is not None:
        st.session_state["canvas_snapshot"] = canvas_image
        st.session_state["canvas_snapshot_size"] = canvas_size
    predict_clicked = st.button("识别数字 / Recognize", type="primary", use_container_width=True)

with right:
    st.subheader("识别结果 | Prediction")
    if predict_clicked:
        if canvas_image is None:
            st.warning("请先在画布中写一个数字。Draw a number first.")
        else:
            result = predict_canvas_number(model, canvas_image, device)
            if result is None:
                st.warning("没有检测到有效笔迹。No visible strokes detected.")
            else:
                prediction, confidence, digit_rows, regions = result
                st.session_state["last_prediction"] = prediction
                st.session_state["last_confidence"] = confidence
                st.session_state["last_digit_rows"] = digit_rows
                st.session_state["last_regions"] = regions
                st.session_state["last_image"] = canvas_image
                if prediction > 999:
                    st.warning("当前页面按 0-999 设计，识别结果超过范围；可以清除后重新书写。")

    if "last_prediction" in st.session_state:
        prediction = st.session_state["last_prediction"]
        confidence = st.session_state["last_confidence"]
        metric_left, metric_right = st.columns(2)
        metric_left.metric("预测数字 | Number", str(prediction))
        metric_right.metric("平均置信度 | Avg. confidence", f"{confidence * 100:.1f}%")

        st.caption("逐位概率详情已移到下方宽区域。Digit-level details are shown below.")
    else:
        st.write("等待识别结果。Waiting for prediction.")

st.divider()
st.subheader("逐位识别详情 | Digit-Level Details")
if "last_prediction" in st.session_state:
    digit_rows = st.session_state["last_digit_rows"]
    detail_rows = pd.DataFrame(
        {
            "position": [row["position"] for row in digit_rows],
            "digit": [str(row["digit"]) for row in digit_rows],
            "confidence": [row["confidence"] for row in digit_rows],
        }
    )
    st.dataframe(detail_rows, hide_index=True, use_container_width=True)

    if len(digit_rows) == 1:
        probability_frame = pd.DataFrame(
            {
                "digit": list(range(10)),
                "probability": digit_rows[0]["probabilities"],
            }
        ).set_index("digit")
        st.bar_chart(probability_frame)
    else:
        tabs = st.tabs([f"第 {row['position']} 位 / Digit {row['position']}" for row in digit_rows])
        for tab, row in zip(tabs, digit_rows):
            with tab:
                probability_frame = pd.DataFrame(
                    {
                        "digit": list(range(10)),
                        "probability": row["probabilities"],
                    }
                ).set_index("digit")
                st.bar_chart(probability_frame)
else:
    st.caption("完成一次识别后显示逐位概率。Run a prediction to inspect digit-level probabilities.")

st.divider()
st.subheader("反馈闭环 | Feedback Loop")
st.markdown(
    "如果模型识别错误，选择或输入真实数字并保存反馈。系统会把图片和标签写入 `feedback/`。"
    "点击更新模型后会离线重训练并直接覆盖当前 `digit_classifier.pt`，不会保留旧模型备份。"
)
with st.expander("模型更新判断标准 | Model Update Criteria", expanded=False):
    st.markdown(
        "- 进度按训练批次累计计算，约以 1% 为单位刷新，并显示预计剩余时间。\n"
        "- 页面实时显示当前 loss 与 accuracy，loss 越低、accuracy 越高越好。\n"
        "- 更新完成后读取 `outputs/metrics.json`，展示测试准确率、反馈样本数和训练耗时。\n"
        "- 当前展示阈值为 MNIST 测试准确率 98%；低于阈值也会覆盖旧模型，但页面会提醒继续积累反馈样本。\n"
        "- 只有真实标签为 0-9 的单数字反馈会直接参与当前 CNN 重训练；两位数和三位数反馈会先作为产品反馈数据保存。"
    )
feedback_total, feedback_trainable = count_feedback_rows(labels_path)
min_feedback_before_update = int(config.get("outputs", {}).get("min_feedback_before_update", 100))
st.caption(
    f"反馈样本 | Feedback samples: {feedback_total}    "
    f"可直接用于单数字训练 | Trainable single-digit samples: {feedback_trainable}"
)
auto_update = st.toggle(
    "达到样本门槛后自动离线更新 / Auto update after enough feedback",
    value=False,
    disabled=feedback_trainable < min_feedback_before_update,
)
if feedback_trainable < min_feedback_before_update:
    st.info(
        f"建议累计至少 {min_feedback_before_update} 条单数字反馈后再更新模型；当前可训练样本为 {feedback_trainable} 条。"
    )

if "last_prediction" in st.session_state:
    prediction = st.session_state["last_prediction"]
    confidence = st.session_state["last_confidence"]
    actual_key = f"actual_label_{st.session_state.get('canvas_version', 0)}_{prediction}"
    if actual_key not in st.session_state:
        st.session_state[actual_key] = str(min(max(int(prediction), 0), 999))
    feedback_left, feedback_right = st.columns([1, 1])
    with feedback_left:
        actual_value = st.text_input(
            "真实数字 | Actual number",
            key=actual_key,
            placeholder="直接输入 0-999，输入后可直接保存",
            help="这里以输入框内容为准，不需要再从下拉结果里点一次。",
        )
        quick_cols = st.columns(10)
        for value, quick_col in enumerate(quick_cols):
            quick_col.button(
                str(value),
                key=f"quick_label_{value}_{actual_key}",
                use_container_width=True,
                on_click=set_actual_label,
                args=(actual_key, value),
            )
    with feedback_right:
        st.write("")
        st.write("")
        if st.button("保存反馈 / Save Feedback", use_container_width=True):
            try:
                normalized_label = normalize_actual_label(actual_value)
                image_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.png"
                st.session_state["last_image"].save(images_dir / image_name)
                append_feedback(labels_path, image_name, str(prediction), normalized_label, confidence)
                st.success(f"反馈已保存为 {normalized_label}。Feedback saved as {normalized_label}.")
                if auto_update:
                    updated_trainable = feedback_trainable + (1 if len(normalized_label) == 1 else 0)
                    if updated_trainable >= min_feedback_before_update:
                        try:
                            run_model_update_ui(metrics_path, updated_trainable)
                        except Exception as exc:
                            st.error(f"模型更新失败 / Model update failed: {exc}")
                    else:
                        st.info("反馈已保存；样本量不足，暂不触发自动训练。")
            except ValueError:
                st.error("请输入 0-999 的有效数字。Please enter a valid number from 0 to 999.")
else:
    st.caption("完成一次识别后即可提交反馈。Run a prediction before submitting feedback.")

if st.button("立即离线更新模型 / Update Model Now", use_container_width=True):
    if feedback_trainable < min_feedback_before_update:
        st.warning(
            f"当前只有 {feedback_trainable} 条可训练反馈，低于建议门槛 {min_feedback_before_update} 条；"
            "本次仍会执行离线训练并发布为新模型版本。"
        )
    try:
        run_model_update_ui(metrics_path, feedback_trainable)
    except Exception as exc:
        st.error(f"模型更新失败 / Model update failed: {exc}")

st.caption(f"模型 | Model: {checkpoint.get('model_type', 'unknown')}    设备 | Device: {device}")
