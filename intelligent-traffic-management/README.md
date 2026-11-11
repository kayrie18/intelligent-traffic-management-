# Intelligent Traffic Management System

CP 81 Comprehension 2, Project 4. Ekari Emmanuel Moyo, Reg. No. 23311351007.

An AI-based prototype that reads simulated vehicle volumes at a four-way junction and recommends
how long each direction should get a green light. A neural network picks the green split every
90-second cycle, and a second-by-second simulation compares it with a fixed 40 s / 40 s plan.

**Portfolio website:** https://kayrie18.github.io/intelligent-traffic-management/
**Repository:** https://github.com/kayrie18/intelligent-traffic-management

## Results (5 simulated days, ~16,800 vehicles per day)

| Controller | Avg delay per vehicle | Peak-hour delay | Longest queue |
|---|---|---|---|
| Fixed-time (40 s / 40 s) | 20.2 s | 23.3 s | 23.4 |
| AI adaptive controller | 17.9 s | 19.3 s | 17.0 |
| Rule with perfect information | 17.7 s | 19.2 s | 16.2 |

The AI controller cuts average delay by about 11% and peak-hour delay by about 17%.
On held-out test data the model is 2.75 s away from the ideal green time on average (R² 0.86).

## How it works

1. `src/common.py` defines the junction (90 s cycle, 80 s of green, 1,800 veh/h saturation flow) and the simulated daily demand.
2. `src/generate_data.py` writes `data/demand_profile.csv` and `data/traffic_training_data.csv` (noisy detector readings plus the ideal green time).
3. `src/train_model.py` trains a 4-16-16-1 neural network and exports `models/model.json`.
4. `src/simulate.py` runs the junction second by second for 24 hours with fixed, AI and reference controllers.
5. `src/run_simulation.py` runs five random days, saves `results/simulation_results.json`, the charts, and the data used by the website.

The labels come from the equal degree-of-saturation (Webster-style) rule, so the network learns to
approximate that rule from noisy readings. It does not discover a better rule than that.

## Run it

```
pip install -r requirements.txt
python run_all.py
```

Open `docs/index.html` in a browser to view the site locally.

## Project layout

```
src/         simulation, data generation, model training
data/        demand profile and training dataset (CSV)
models/      trained model weights (JSON)
results/     metrics and simulation results (JSON)
docs/        three-page portfolio website (served by GitHub Pages)
run_all.py   regenerates everything
```

## Limitations

Simulated data and a simplified junction: one lane per approach, two phases, no turns, pedestrians,
buses or neighbouring junctions. A real deployment would need measured counts and validation first.
