"""Train the green-time model (pure NumPy, no scikit-learn/SciPy) and export it as JSON."""
import json
import numpy as np
import pandas as pd
from pathlib import Path

from .common import APPROACHES, GREEN_TOTAL, FIXED_NS, MIN_GREEN

ROOT = Path(__file__).resolve().parent.parent
IN_SCALE, OUT_SCALE = 1000.0, float(GREEN_TOTAL)
LAYERS = [4, 16, 16, 1]
EPOCHS, BATCH, LR = 250, 128, 0.003


def forward(params, X):
    acts = [X]
    for i, (W, b) in enumerate(params):
        z = acts[-1] @ W + b
        acts.append(np.maximum(z, 0.0) if i < len(params) - 1 else z)
    return acts


def train(X, y, seed=7):
    rng = np.random.default_rng(seed)
    params = [(rng.normal(0, np.sqrt(2.0 / a), (a, b)), np.zeros(b)) for a, b in zip(LAYERS[:-1], LAYERS[1:])]
    m = [(np.zeros_like(W), np.zeros_like(b)) for W, b in params]
    v = [(np.zeros_like(W), np.zeros_like(b)) for W, b in params]
    step = 0
    for _ in range(EPOCHS):
        order = rng.permutation(len(X))
        for s in range(0, len(X), BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx].reshape(-1, 1)
            acts = forward(params, xb)
            delta = 2 * (acts[-1] - yb) / len(xb)          # d(MSE)/d(output)
            step += 1
            for i in range(len(params) - 1, -1, -1):
                W, b = params[i]
                gW, gb = acts[i].T @ delta, delta.sum(0)
                if i > 0:
                    delta = (delta @ W.T) * (acts[i] > 0)
                for k, g in ((0, gW), (1, gb)):               # Adam update
                    m[i] = tuple(0.9 * m[i][j] + 0.1 * g if j == k else m[i][j] for j in (0, 1))
                    v[i] = tuple(0.999 * v[i][j] + 0.001 * g * g if j == k else v[i][j] for j in (0, 1))
                    mh = m[i][k] / (1 - 0.9 ** step)
                    vh = v[i][k] / (1 - 0.999 ** step)
                    p = params[i][k] - LR * mh / (np.sqrt(vh) + 1e-8)
                    params[i] = (p, params[i][1]) if k == 0 else (params[i][0], p)
    return params


def main():
    df = pd.read_csv(ROOT / "data" / "traffic_training_data.csv")
    X = df[[f"sensor_{a.lower()}_vph" for a in APPROACHES]].values / IN_SCALE
    y = df["ideal_ns_green_s"].values / OUT_SCALE
    perm = np.random.default_rng(1).permutation(len(X))
    cut = int(len(X) * 0.8)
    tr, te = perm[:cut], perm[cut:]

    params = train(X[tr], y[tr])
    pred = np.clip(forward(params, X[te])[-1].ravel() * OUT_SCALE, MIN_GREEN, GREEN_TOTAL - MIN_GREEN)
    truth = y[te] * OUT_SCALE
    ss_res, ss_tot = ((truth - pred) ** 2).sum(), ((truth - truth.mean()) ** 2).sum()
    metrics = {
        "train_rows": int(len(tr)), "test_rows": int(len(te)),
        "mae_seconds": round(float(np.abs(truth - pred).mean()), 2),
        "r2": round(float(1 - ss_res / ss_tot), 3),
        "fixed_plan_mae_seconds": round(float(np.abs(truth - FIXED_NS).mean()), 2),
    }
    spec = {
        "input_scale": IN_SCALE, "output_scale": OUT_SCALE,
        "weights": [W.round(6).tolist() for W, _ in params],
        "biases": [b.round(6).tolist() for _, b in params],
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
