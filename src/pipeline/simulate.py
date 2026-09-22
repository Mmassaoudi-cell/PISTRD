"""Generates the TRUE (attack-free) state/measurement trajectory for a given
IEEE bus testbed, driven by real PJM hourly load data. DC power flow is linear
so the trajectory depends only on the load-multiplier series, never on
noise/attack realizations -- computed once per testbed and reused, exactly as
the DC-motor reproduction reused a single deterministic true trajectory
across Monte-Carlo repetitions (see cps_qlearning.py)."""
from pathlib import Path
import numpy as np
import pandas as pd
import pandapower as pp

from testbeds import get_testbed, get_topology_edges

LOAD_CSV = r"C:\Users\MMASSAOUDI\Desktop\Data\Load Data\PJM_Load_hourly.csv"


def load_multiplier_series(T: int, start_offset: int = 0) -> np.ndarray:
    df = pd.read_csv(LOAD_CSV, parse_dates=["Datetime"]).sort_values("Datetime").reset_index(drop=True)
    series = df["PJM_Load_MW"].values.astype(float)
    series = series[start_offset:start_offset + T]
    mult = series / series.mean()
    return mult, df["Datetime"].values[start_offset:start_offset + T]


def simulate_true_trajectory(testbed_name: str, T: int, start_offset: int = 0,
                              mult: np.ndarray = None, timestamps=None):
    """If `mult` is provided (a length-T load-multiplier array), it overrides
    the default PJM series -- used for cross-load-source generalization
    testing (see src/generalization/cross_load_source.py)."""
    net = get_testbed(testbed_name)
    edges = get_topology_edges(net)
    base_p = net.load.p_mw.values.copy()
    base_q = net.load.q_mvar.values.copy()
    if mult is None:
        mult, timestamps = load_multiplier_series(T, start_offset)

    n_bus = len(net.bus)
    n_line = len(net.line)
    va = np.zeros((T, n_bus))
    line_flow = np.zeros((T, n_line))

    for t in range(T):
        net.load.p_mw = base_p * mult[t]
        net.load.q_mvar = base_q * mult[t]
        pp.rundcpp(net)
        va[t] = net.res_bus.va_degree.values
        line_flow[t] = net.res_line.p_from_mw.values

    return {
        "testbed": testbed_name,
        "n_bus": n_bus,
        "n_line": n_line,
        "edges": edges,
        "va": va,             # (T, n_bus) -- state analogue
        "line_flow": line_flow,  # (T, n_line) -- measurement analogue (y_k)
        "timestamps": timestamps,
        "load_multiplier": mult,
    }


if __name__ == "__main__":
    import time
    for tb in ("case14", "case39", "case57", "case118"):
        t0 = time.time()
        out = simulate_true_trajectory(tb, T=3000)
        print(tb, "n_bus", out["n_bus"], "n_line", out["n_line"],
              "edges", out["edges"].shape, "time", round(time.time() - t0, 1), "s",
              "line_flow range", out["line_flow"].min(), out["line_flow"].max())
