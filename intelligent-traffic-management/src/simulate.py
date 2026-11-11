"""Second-by-second simulation of a four-way signalised junction over one day."""
import numpy as np

from .common import (CYCLE, GREEN_TOTAL, SAT_VPS, FIXED_NS, demand_vph, ideal_green_ns)
from .ai_model import GreenTimeModel

T_DAY = 24 * 3600
AMBER = 5          # seconds of amber / all-red after each green
EMA_ALPHA = 0.35   # smoothing of the detector counts


class FixedController:
    name = "Fixed-time (40 s / 40 s)"

    def decide(self, counts, t_mid):
        return FIXED_NS


class WebsterController:
    """Reference: same rule used for the training labels, but fed the TRUE volumes."""
    name = "Rule with perfect information"

    def decide(self, counts, t_mid):
        return int(round(float(ideal_green_ns(demand_vph(t_mid / 3600.0)))))


class AIController:
    """Neural network fed with smoothed detector counts from the previous cycles."""
    name = "AI adaptive controller"

    def __init__(self):
        self.model = GreenTimeModel()
        self.ema = None

    def decide(self, counts, t_mid):
        if counts is None:
            return FIXED_NS
        vph = np.asarray(counts, dtype=float) * 3600.0 / CYCLE
        self.ema = vph if self.ema is None else EMA_ALPHA * vph + (1 - EMA_ALPHA) * self.ema
        return self.model.predict_ns_green(self.ema)


def simulate_day(controller, seed=0):
    rng = np.random.default_rng(seed)
    t_axis = np.arange(T_DAY) + 0.5
    arrivals = rng.poisson(demand_vph(t_axis / 3600.0) / 3600.0)   # (T, 4) vehicles per second

    queue = np.zeros(4)
    credit = np.zeros(4)
    q_seconds = np.zeros((24, 4))
    arrived = np.zeros((24, 4))
    max_queue = np.zeros(4)
    g_log = []
    counts = None

    for c in range(T_DAY // CYCLE):
        t0 = c * CYCLE
        g_ns = int(controller.decide(counts, t0 + CYCLE / 2))
        g_ns = min(max(g_ns, 10), GREEN_TOTAL - 10)
        g_ew = GREEN_TOTAL - g_ns
        g_log.append(g_ns)
        # seconds within the cycle: NS green, amber, EW green, amber
        ns_end, ew_start = g_ns, g_ns + AMBER
        ew_end = ew_start + g_ew
        cycle_counts = np.zeros(4)
        for s in range(CYCLE):
            t = t0 + s
            hr = t // 3600
            queue += arrivals[t]
            cycle_counts += arrivals[t]
            arrived[hr] += arrivals[t]
            green = (0, 1) if s < ns_end else ((2, 3) if ew_start <= s < ew_end else ())
            for i in green:
                credit[i] += SAT_VPS
                while credit[i] >= 1.0 and queue[i] >= 1.0:
                    queue[i] -= 1.0
                    credit[i] -= 1.0
                if queue[i] < 1.0:
                    credit[i] = 0.0
            for i in range(4):
                if i not in green:
                    credit[i] = 0.0
            q_seconds[hr] += queue
            max_queue = np.maximum(max_queue, queue)
        counts = cycle_counts

    total_arr = arrived.sum()
    delay_hour = q_seconds.sum(1) / np.maximum(arrived.sum(1), 1)
    return {
        "avg_delay_s": float(q_seconds.sum() / total_arr),
        "delay_by_hour": delay_hour,
        "max_queue": float(max_queue.max()),
        "mean_queue": float(q_seconds.sum() / T_DAY / 4),
        "vehicles": int(total_arr),
        "ns_green": np.array(g_log),
        "final_queue": float(queue.sum()),
    }
