"""Train the green-time model and export it as JSON (used by Python and the website)."""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from .common import APPROACHES, GREEN_TOTAL, FIXED_NS, MIN_GREEN

ROOT = Path(__file__).resolve().parent.parent
IN_SCALE, OUT_SCALE = 1000.0, float(GREEN_TOTAL)


def main():
    df = pd.read_csv(ROOT / "data" / "traffic_training_data.csv")
    X = df[[f"sensor_{a.lower()}_vph" for a in APPROACHES]].values / IN_SCALE
    y = df["ideal_ns_green_s"].values / OUT_SCALE
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=1)

    net = MLPRegressor(hidden_layer_sizes=(16, 16), activation="relu", max_iter=800,
                       early_stopping=True, random_state=7)
    net.fit(X_tr, y_tr)

    pred = np.clip(net.predict(X_te) * OUT_SCALE, MIN_GREEN, GREEN_TOTAL - MIN_GREEN)
    truth = y_te * OUT_SCALE
    metrics = {
        "train_rows": int(len(X_tr)), "test_rows": int(len(X_te)),
        "mae_seconds": round(float(mean_absolute_error(truth, pred)), 2),
        "r2": round(float(r2_score(truth, pred)), 3),
        "fixed_plan_mae_seconds": round(float(mean_absolute_error(truth, np.full_like(truth, FIXED_NS))), 2),
    }
    spec = {
        "input_scale": IN_SCALE, "output_scale": OUT_SCALE,
        "weights": [w.round(6).tolist() for w in net.coefs_],
        "biases": [b.round(6).tolist() for b in net.intercepts_],
        "metrics": metrics,
    }
    (ROOT / "models").mkdir(exist_ok=True)
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "models" / "model.json").write_text(json.dumps(spec))
    (ROOT / "results" / "model_metrics.json").write_text(json.dumps(metrics, indent=2))
    np.save(ROOT / "results" / "_scatter.npy", np.vstack([truth, pred]))
    print("model metrics:", metrics)


if __name__ == "__main__":
    main()
