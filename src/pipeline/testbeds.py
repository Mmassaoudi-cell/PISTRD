"""IEEE bus-system testbed loading + topology extraction for the multi-testbed
power-system replay-attack extension (see DATA_AUDIT.md)."""
import numpy as np
import pandapower as pp
import pandapower.networks as pn

TESTBEDS = ("case14", "case39", "case57", "case118")


def get_testbed(name: str):
    return {"case14": pn.case14, "case39": pn.case39,
            "case57": pn.case57, "case118": pn.case118}[name]()


def get_topology_edges(net) -> np.ndarray:
    """Undirected edge list (2,E) over bus indices, combining lines + transformers."""
    edges = []
    for _, row in net.line.iterrows():
        edges.append((int(row.from_bus), int(row.to_bus)))
    if len(net.trafo) > 0:
        for _, row in net.trafo.iterrows():
            edges.append((int(row.hv_bus), int(row.lv_bus)))
    edges = np.array(edges, dtype=np.int64).T  # (2, E)
    return edges


def n_buses(net) -> int:
    return len(net.bus)
