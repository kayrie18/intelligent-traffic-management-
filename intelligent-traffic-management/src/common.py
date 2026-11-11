"""Shared constants, the simulated demand model and the signal-timing rule."""
import numpy as np

APPROACHES = ["North", "South", "East", "West"]

# Junction assumptions: one lane per approach, two phases (North-South, East-West)
CYCLE = 90                      # seconds per signal cycle
LOST_TIME = 10                  # 2 x 5 s amber / all-red per cycle
GREEN_TOTAL = CYCLE - LOST_TIME  # 80 s of green to share between the two phases
MIN_GREEN = 10                  # no phase may get less than this
SAT_VPH = 1800.0                # saturation flow: vehicles per hour of green, per lane
SAT_VPS = SAT_VPH / 3600.0
FIXED_NS = GREEN_TOTAL // 2     # baseline: 40 s / 40 s

PEAK = 650.0                    # scales the simulated demand (vehicles per hour)


def _bump(h, centre, width):
    return np.exp(-((h - centre) / width) ** 2)


def demand_vph(hour):
    """Simulated vehicle volumes (veh/h) for North, South, East, West at a given hour.

    Morning peak favours the North approach (inbound commuters), the evening peak
    favours South, and a lunchtime market surge loads East and West.
    `hour` may be a scalar or an array; the result has shape (..., 4).
    """
    h = np.asarray(hour, dtype=float)[..., None]
    night = np.array([0.5, 0.5, 0.4, 0.4])
    daytime = np.array([0.6, 0.6, 0.5, 0.5])
    am_w = np.array([0.95, 0.50, 0.40, 0.30])
    pm_w = np.array([0.50, 0.95, 0.30, 0.40])
    lunch_w = np.array([0.00, 0.00, 0.75, 0.55])
    v = (0.15 * night
         + 0.35 * _bump(h, 13, 5.5) * daytime
         + _bump(h, 8, 1.3) * am_w
         + _bump(h, 17, 1.4) * pm_w
         + _bump(h, 12.5, 0.7) * lunch_w)
    return PEAK * v


def ideal_green_ns(vols):
    """Equal degree-of-saturation rule (Webster-style).

    Each phase is limited by its busiest approach (critical flow ratio y = v / s).
    Green time is shared in proportion to y, so both phases are equally loaded.
    `vols` has shape (..., 4) ordered North, South, East, West.
    """
    vols = np.asarray(vols, dtype=float)
    y_ns = np.maximum(vols[..., 0], vols[..., 1]) / SAT_VPH
    y_ew = np.maximum(vols[..., 2], vols[..., 3]) / SAT_VPH
    total = y_ns + y_ew
    share = np.where(total > 0, y_ns / np.where(total > 0, total, 1.0), 0.5)
    return np.clip(GREEN_TOTAL * share, MIN_GREEN, GREEN_TOTAL - MIN_GREEN)
