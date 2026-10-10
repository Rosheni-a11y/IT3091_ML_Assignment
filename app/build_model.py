"""
Rebuild app/model.joblib from the model the notebook saved.

The notebook writes artifacts/final_model.joblib at ~44 MB, which is heavier
than it needs to be for a deployed app. joblib's compression takes it to ~12 MB
with bit-identical predictions, which is small enough to commit.

Run from the repository root, after 02_Model_Training.ipynb:

    python app/build_model.py
"""
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "artifacts" / "final_model.joblib"
DST = ROOT / "app" / "model.joblib"

if not SRC.exists():
    raise SystemExit(f"{SRC} not found - run 02_Model_Training.ipynb first.")

model = joblib.load(SRC)
joblib.dump(model, DST, compress=6)

# prove the compressed copy predicts identically
manifest = json.loads((ROOT / "artifacts" / "manifest.json").read_text())
features = manifest["categorical_features"] + manifest["numeric_features"]
test = pd.read_parquet(ROOT / "artifacts" / "test.parquet")

a = roc_auc_score(test[manifest["target"]], model.predict_proba(test[features])[:, 1])
b = roc_auc_score(test[manifest["target"]],
                  joblib.load(DST).predict_proba(test[features])[:, 1])

print(f"source      {SRC.stat().st_size / 1048576:6.1f} MB   ROC-AUC {a:.6f}")
print(f"compressed  {DST.stat().st_size / 1048576:6.1f} MB   ROC-AUC {b:.6f}")
print("identical predictions:", a == b)
