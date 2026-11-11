"""Generate the simulated datasets used by the project.

data/demand_profile.csv        - hourly vehicle volumes for one simulated day
data/traffic_training_data.csv - sensor readings (noisy volumes) with the ideal NS green time
"""
import numpy as np
import pandas as pd
from pathlib import Path

from .common import APPROACHES, demand_vph, ideal_green_ns

DATA = Path(__file__).resolve().parent.parent / "data"
SENSOR_NOISE = 0.18   # relative error of a detector reading taken over a short window


def make_training_data(n=15000, seed=42):
    rng = np.random.default_rng(seed)
    n_profile = int(n * 0.7)
    hours = rng.uniform(0, 24, n_profile)
    scale = rng.uniform(0.6, 1.35, (n_profile, 1))
    jitter = rng.uniform(0.75, 1.25, (n_profile, 4))
    profile_vols = demand_vph(hours) * scale * jitter
    uniform_vols = rng.uniform(20, 900, (n - n_profile, 4))
    true_vols = np.clip(np.vstack([profile_vols, uniform_vols]), 5, 950)
    noisy = true_vols * rng.lognormal(0.0, SENSOR_NOISE, true_vols.shape)
    df = pd.DataFrame(np.round(noisy, 1), columns=[f"sensor_{a.lower()}_vph" for a in APPROACHES])
    for i, a in enumerate(APPROACHES):
        df[f"true_{a.lower()}_vph"] = np.round(true_vols[:, i], 1)
    df["ideal_ns_green_s"] = np.round(ideal_green_ns(true_vols), 2)
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def main():
    DATA.mkdir(exist_ok=True)
    hours = np.arange(24) + 0.5
    prof = pd.DataFrame(np.round(demand_vph(hours)).astype(int), columns=[f"{a.lower()}_vph" for a in APPROACHES])
    prof.insert(0, "hour", np.arange(24))
    prof.to_csv(DATA / "demand_profile.csv", index=False)
    make_training_data().to_csv(DATA / "traffic_training_data.csv", index=False)
    print("wrote data/demand_profile.csv and data/traffic_training_data.csv")


if __name__ == "__main__":
    main()
