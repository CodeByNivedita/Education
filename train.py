"""Optional check: trains the same model the app uses and prints a summary.
The app trains by itself on first start, so you do NOT need to run this.

Usage:  python train.py [path/to/student.csv]
"""
import sys

from model_utils import train

csv = sys.argv[1] if len(sys.argv) > 1 else "student.csv"
model, meta = train(csv)
print("File:", csv, "| rows:", meta["rows"])
print("Model:", meta["model_name"], "| sanity check passed:", meta["sanity_check_passed"])
print("Inputs used:", meta["features"])
print("GPA range in data:", round(meta["target_min"], 3), "to", round(meta["target_max"], 3))
print("Accuracy:", {k: round(v, 4) for k, v in meta["metrics"].items()})
print("Test predictions range:", round(meta["pred_min"], 3), "to", round(meta["pred_max"], 3),
      "(mean", round(meta["pred_mean"], 3), ")")