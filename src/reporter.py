from pathlib import Path


def build_report(model_type, epochs, metrics, report_path: Path, include_training_curve: bool = False):
    class_rows = "\n".join(
        f"<tr><td>{digit}</td><td>{accuracy:.3f}</td></tr>"
        for digit, accuracy in metrics["class_accuracy"].items()
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
    <h2>Accuracy</h2>
    <div class="metric">{metrics["accuracy"]:.3f}</div>
  </section>
  <section class="panel">
    <h2>Class Accuracy</h2>
    <table border="1" cellspacing="0" cellpadding="8">
      <tr><th>Digit</th><th>Accuracy</th></tr>
      {class_rows}
    </table>
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
