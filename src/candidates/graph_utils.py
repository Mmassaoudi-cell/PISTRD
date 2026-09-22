"""Builds the 'line graph' (dual graph) needed by STGAT: our measurements are
per-LINE (edges of the electrical network), but a GAT needs a node graph, so
each line becomes a node, connected to other lines that share a bus in the
original topology. Only the first n_line columns of the combined edges array
(from testbeds.get_topology_edges, which appends transformer edges after line
edges) are used to build this adjacency -- transformer connectivity is not
included as an extra graph node since it is not part of the measured stream
(documented simplification, see MODEL_SELECTION_REPORT.md)."""
import numpy as np


def build_line_graph(edges: np.ndarray, n_line: int) -> np.ndarray:
    line_edges = edges[:, :n_line]  # (2, n_line) from_bus/to_bus per line
    bus_to_lines = {}
    for li in range(n_line):
        for b in (int(line_edges[0, li]), int(line_edges[1, li])):
            bus_to_lines.setdefault(b, []).append(li)
    pairs = set()
    for b, lines in bus_to_lines.items():
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                a, c = lines[i], lines[j]
                pairs.add((min(a, c), max(a, c)))
    if not pairs:
        # fallback: fully disconnected -> self loops only
        idx = np.arange(n_line)
        return np.stack([idx, idx])
    src, dst = zip(*pairs)
    src, dst = list(src), list(dst)
    # make undirected + self loops
    ei = np.array([src + dst + list(range(n_line)),
                    dst + src + list(range(n_line))], dtype=np.int64)
    return ei
