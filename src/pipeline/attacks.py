"""Randomized generalization of the source paper's 3 replay-attack strategies
(continuous / periodic / aperiodic) to arbitrary time ranges, for building a
diverse labeled dataset instead of 3 fixed scenarios."""
import numpy as np


def generate_scenario(t0: int, t1: int, rng: np.random.Generator,
                       durations=(8, 16, 24, 36), max_tries=20):
    for _ in range(max_tries):
        strategy = int(rng.choice([1, 2, 3]))
        n_intervals = {1: 1, 2: 2, 3: 3}[strategy]
        D = int(rng.choice(durations))
        margin = D * (n_intervals + 2) + 15
        if t1 - t0 <= margin:
            continue
        rec_start = int(rng.integers(t0, t1 - margin))
        rec_end = rec_start + D
        cur = rec_end + int(rng.integers(5, 20))
        attacks = []
        ok = True
        for _ in range(n_intervals):
            a_start = cur
            a_end = a_start + D
            if a_end >= t1:
                ok = False
                break
            attacks.append((a_start, a_end))
            cur = a_end + int(rng.integers(5, 30))
        if ok:
            return {"strategy": strategy, "duration": D,
                    "record": (rec_start, rec_end), "attacks": attacks}
    # fallback: minimal continuous attack near the start of the block
    D = 6
    return {"strategy": 1, "duration": D, "record": (t0, t0 + D),
            "attacks": [(t0 + D + 5, t0 + 2 * D + 5)]}


def scenario_to_labels(scenario: dict, T: int):
    """Returns (replay_of: dict k->recorded_index, label: bool array length T)."""
    replay_of = {}
    label = np.zeros(T, dtype=bool)
    rec_start, rec_end = scenario["record"]
    for a_start, a_end in scenario["attacks"]:
        shift = a_start - rec_start
        for k in range(a_start, a_end):
            ridx = k - shift
            if rec_start <= ridx < rec_end:
                replay_of[k] = ridx
                label[k] = True
    return replay_of, label
