from pathlib import Path


def build_report(model_type, epochs, metrics, report_path: Path, include_training_curve: bool = False):
    class_rows = "\n".join(
        "<tr>"
        f"<td>{digit}</td>"
        f"<td>{metrics['class_accuracy'].get(digit, 0):.3f}</td>"
        f"<td>{row['precision']:.3f}</td>"
        f"<td>{row['recall']:.3f}</td>"
        f"<td>{row['f1_score']:.3f}</td>"
        f"<td>{row['support']}</td>"
        "</tr>"
        for digit, row in metrics.get("per_class", {}).items()
    )
    training_curve_section = ""
    if include_training_curve:
        training_curve_section = """
  <section class="panel">
    <h2>Training Curve</h2>
    <img src="training_curve.png" alt="Training curve">
  </section>
"""
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>DeepVision Studio Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; color: #1f2937; }}
    .panel {{ border: 1px solid #d6dbe5; border-radius: 8px; padding: 18px; margin-bottom: 16px; }}
    .metric {{ font-size: 32px; color: #2563eb; font-weight: bold; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 16px; }}
    img {{ max-width: 760px; width: 100%; border: 1px solid #d6dbe5; border-radius: 8px; }}
  </style>
</head>
<body>
  <h1>DeepVision Studio Report</h1>
  <section class="panel">
    <h2>Model</h2>
    <p>{model_type}</p>
    <p>Epochs: {epochs}</p>
  </section>
  <section class="panel">
    <h2>Evaluation</h2>
    <div class="grid">
      <div><strong>Accuracy</strong><div class="metric">{metrics["accuracy"]:.3f}</div></div>
      <div><strong>Macro Precision</strong><div class="metric">{metrics.get("macro_precision", 0):.3f}</div></div>
      <div><strong>Macro Recall</strong><div class="metric">{metrics.get("macro_recall", 0):.3f}</div></div>
      <div><strong>Macro F1</strong><div class="metric">{metrics.get("macro_f1", 0):.3f}</div></div>
    </div>
    <p>Loss: {metrics.get("loss", 0):.4f}</p>
  </section>
  <section class="panel">
    <h2>Per-Class Metrics</h2>
    <table border="1" cellspacing="0" cellpadding="8">
      <tr><th>Digit</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th></tr>
      {class_rows}
    </table>
  </section>
  <section class="panel">
    <h2>Confusion Matrix</h2>
    <img src="confusion_matrix.png" alt="Test confusion matrix">
  </section>
  <section class="panel">
    <h2>Sample Predictions</h2>
    <img src="sample_predictions.png" alt="Sample predictions">
  </section>
  {training_curve_section}
</body>
</html>
"""
    report_path.parent.mkdir(exist_ok=True)
    report_path.write_text(html, encoding="utf-8")
