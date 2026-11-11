"""Run fixed-time vs AI vs reference controllers over several simulated days; make charts + site data."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from .common import APPROACHES, CYCLE, demand_vph
from .simulate import FixedController, AIController, WebsterController, simulate_day

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "docs" / "assets"
SEEDS = [1, 2, 3, 4, 5]
INK, GREEN, RED, AMBER_C, GREY = "#22303A", "#17855A", "#C23B2E", "#D9930D", "#8A949B"
PEAK_HOURS = [7, 8, 16, 17]


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "axes.edgecolor": GREY, "axes.labelcolor": INK,
        "xtick.color": INK, "ytick.color": INK, "axes.spines.top": False,
        "axes.spines.right": False, "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.grid": True, "grid.color": "#E3E6E1", "grid.linewidth": 0.8,
    })


def main():
    ASSETS.mkdir(parents=True, exist_ok=True)
    style()
    makers = {"fixed": FixedController, "ai": AIController, "rule": WebsterController}
    runs = {k: [simulate_day(m(), s) for s in SEEDS] for k, m in makers.items()}

    summary = {}
    for k, rs in runs.items():
        d = np.array([r["avg_delay_s"] for r in rs])
        ph = np.array([r["delay_by_hour"][PEAK_HOURS].mean() for r in rs])
        summary[k] = {
            "name": makers[k].name,
            "avg_delay_s": round(float(d.mean()), 1), "avg_delay_sd": round(float(d.std()), 1),
            "peak_delay_s": round(float(ph.mean()), 1),
            "max_queue": round(float(np.mean([r["max_queue"] for r in rs])), 1),
            "mean_queue": round(float(np.mean([r["mean_queue"] for r in rs])), 1),
            "vehicles": int(np.mean([r["vehicles"] for r in rs])),
        }
    f, a = summary["fixed"], summary["ai"]
    summary["improvement_avg_pct"] = round(100 * (f["avg_delay_s"] - a["avg_delay_s"]) / f["avg_delay_s"], 1)
    summary["improvement_peak_pct"] = round(100 * (f["peak_delay_s"] - a["peak_delay_s"]) / f["peak_delay_s"], 1)
    summary["seeds"] = SEEDS

    hours = np.arange(24)
    delay_h = {k: np.mean([r["delay_by_hour"] for r in rs], axis=0) for k, rs in runs.items()}
    ns_green = np.mean([r["ns_green"] for r in runs["ai"]], axis=0)
    summary["delay_by_hour"] = {k: [round(float(x), 1) for x in v] for k, v in delay_h.items()}
    (ROOT / "results" / "simulation_results.json").write_text(json.dumps(summary, indent=2))

    # --- chart 1: demand profile
    prof = demand_vph(np.linspace(0, 24, 289)[:-1])
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for i, (c, ls) in enumerate(zip([INK, GREY, RED, AMBER_C], ["-", "-", "-", "-"])):
        ax.plot(np.linspace(0, 24, 289)[:-1], prof[:, i], color=c, lw=2, label=APPROACHES[i], ls=ls)
    ax.set_xlim(0, 24); ax.set_xticks(range(0, 25, 3))
    ax.set_xlabel("Hour of day"); ax.set_ylabel("Vehicles per hour")
    ax.set_title("Simulated demand on each approach", loc="left", fontsize=12, color=INK)
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    fig.tight_layout(); fig.savefig(ASSETS / "demand_profile.png", dpi=170); plt.close(fig)

    # --- chart 2: delay by hour
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.plot(hours, delay_h["fixed"], color=RED, lw=2.2, marker="o", ms=3.5, label="Fixed-time")
    ax.plot(hours, delay_h["ai"], color=GREEN, lw=2.2, marker="o", ms=3.5, label="AI adaptive")
    ax.set_xlim(0, 23); ax.set_xticks(range(0, 24, 3)); ax.set_ylim(bottom=0)
    ax.set_xlabel("Hour of day"); ax.set_ylabel("Average delay per vehicle (s)")
    ax.set_title("Delay per vehicle through the day (mean of 5 simulated days)", loc="left", fontsize=12, color=INK)
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(ASSETS / "delay_by_hour.png", dpi=170); plt.close(fig)

    # --- chart 3: NS green chosen by AI
    fig, ax = plt.subplots(figsize=(8, 3.2))
    t = np.arange(len(ns_green)) * CYCLE / 3600.0
    k = 20
    smooth = np.convolve(ns_green, np.ones(k) / k, mode="same")
    ax.plot(t, smooth, color=GREEN, lw=2)
    ax.axhline(40, color=RED, lw=1.5, ls="--"); ax.text(23.8, 41.2, "fixed plan: 40 s", color=RED, ha="right", fontsize=9)
    ax.set_xlim(0, 24); ax.set_xticks(range(0, 25, 3)); ax.set_ylim(10, 70)
    ax.set_xlabel("Hour of day"); ax.set_ylabel("North-South green (s)")
    ax.set_title("Green time the AI gives the North-South phase", loc="left", fontsize=12, color=INK)
    fig.tight_layout(); fig.savefig(ASSETS / "ns_green_by_hour.png", dpi=170); plt.close(fig)

    # --- chart 4: model accuracy
    truth, pred = np.load(ROOT / "results" / "_scatter.npy")
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.scatter(truth, pred, s=5, alpha=0.25, color=GREEN, edgecolors="none")
    ax.plot([10, 70], [10, 70], color=INK, lw=1.2)
    ax.set_xlabel("Ideal NS green (s)"); ax.set_ylabel("Model prediction (s)")
    ax.set_title("Model vs ideal, test set", loc="left", fontsize=12, color=INK)
    fig.tight_layout(); fig.savefig(ASSETS / "model_accuracy.png", dpi=170); plt.close(fig)

    # --- data.js for the website (model + hourly profile + results)
    model = json.loads((ROOT / "models" / "model.json").read_text())
    profile = pd.read_csv(ROOT / "data" / "demand_profile.csv")
    payload = {"model": model, "profile": profile[[f"{a.lower()}_vph" for a in APPROACHES]].values.tolist(),
               "results": summary}
    (ASSETS / "data.js").write_text("window.TRAFFIC = " + json.dumps(payload) + ";\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "delay_by_hour"}, indent=1))


if __name__ == "__main__":
    main()
