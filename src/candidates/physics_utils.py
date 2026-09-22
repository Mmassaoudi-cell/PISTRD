"""Extracts the DC power-flow observation matrix relating bus voltage angles
to LINE flows specifically (excluding transformers, which are not part of
our measured stream) -- verified empirically that PandaPower's internal
`Bf` matrix's first n_line rows, in order, exactly reconstruct
net.res_line.p_from_mw (see conversation record / DATA_AUDIT.md note)."""
import numpy as np
import pandapower as pp
from testbeds import get_testbed


def get_line_observation_matrix(testbed_name: str):
    net = get_testbed(testbed_name)
    pp.rundcpp(net)
    ppc = net._ppc
    Bf = ppc["internal"]["Bf"]
    Bf = Bf.toarray() if hasattr(Bf, "toarray") else np.asarray(Bf)
    n_line = len(net.line)
    baseMVA = ppc["baseMVA"]
    return Bf[:n_line, :] * baseMVA  # so that z_MW = H_line @ Va_radians
