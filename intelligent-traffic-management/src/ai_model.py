"""Small neural network (NumPy forward pass) shared by the simulation and the web demo."""
import json
import numpy as np
from pathlib import Path

from .common import MIN_GREEN, GREEN_TOTAL

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "model.json"


class GreenTimeModel:
    def __init__(self, path=MODEL_PATH):
        spec = json.loads(Path(path).read_text())
        self.weights = [np.array(w) for w in spec["weights"]]
        self.biases = [np.array(b) for b in spec["biases"]]
        self.in_scale = spec["input_scale"]
        self.out_scale = spec["output_scale"]

    def predict_ns_green(self, vols):
        x = np.asarray(vols, dtype=float) / self.in_scale
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            x = x @ w + b
            if i < len(self.weights) - 1:
                x = np.maximum(x, 0.0)
        g = float(np.squeeze(x)) * self.out_scale
        return int(round(min(max(g, MIN_GREEN), GREEN_TOTAL - MIN_GREEN)))
