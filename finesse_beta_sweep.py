#!/usr/bin/env python3
"""
Beta-sensitivity sweep, consistent with finesse_transpile.

For a curated set of circuits, records the best achievable log-infidelity cost
at each beta on the grid, using the exact finesse_transpile setup (consolidated
circuit, bidir-warmed layouts + VF2 candidate, deployed beta grid). This is the
landscape the search navigates: the minimum of each curve is what
finesse_transpile returns.

Output: Results/beta_sweep_<tag>.csv with columns
    device, circuit, beta, lf_cost
(beta = 'inf' for the pure-fidelity trial). The Qiskit SABRE baseline for the
same circuits is in Results/qiskit_sabre_<tag>.csv.

Usage:
    python3 finesse_beta_sweep.py --jobs 24            # SNAIL
    python3 finesse_beta_sweep.py --ibm --jobs 24      # IBM fake_brisbane
"""
import argparse
import csv
import math
import os
import warnings
from concurrent.futures import ProcessPoolExecutor

from FrequencyAllocationRuns import (
    build_paper_circuits, build_topology, build_ibm_topologies,
)
from finesse import finesse_transpile

# ~6 circuits spanning sizes and routing demand.
CIRCUITS = ['seca_n11', 'qft_n10', 'multiplier_n15', 'bv_n19', 'qft_n24', 'square_root_n18']
# Sensitivity figure needs a DENSE beta grid (the deployed default is coarse).
DENSE_BETAS = [0, 0.1, 0.2, 0.3, 0.5, 1, 2, 3, 5, 7,
               10, 20, 30, 50, 100, 150, 200, 300, 500, 1000]
# More seeds than deployed for a smooth mean(beta). n_traversals=2 as deployed.
N_SEEDS = 8
N_TRAVERSALS = 2
SEED = 0

_DEVS = None
_CIRCS = None


def _init(tag):
    global _DEVS, _CIRCS
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _DEVS = (build_ibm_topologies() if tag == "ibm"
                 else build_topology(wraparound=False))
        _CIRCS = {n: qc for n, qc in build_paper_circuits()}


def _work(item):
    dev_idx, circ_name = item
    dev_name, cm, F, basis_gate = _DEVS[dev_idx]
    qc = _CIRCS[circ_name]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # use_vf2=False: sensitivity is about routing vs beta; VF2 is a fixed
        # layout that would mask the beta dependence on embeddable circuits.
        _, _, curve = finesse_transpile(
            qc, cm, F, basis_gate=basis_gate, betas=DENSE_BETAS,
            n_seeds=N_SEEDS, n_traversals=N_TRAVERSALS, seed=SEED,
            parallel=False, use_vf2=False, consolidate=True, return_curve=True,
        )
    # one row per (beta, seed): curve is [(beta, [lf per warmed seed])]
    rows = []
    for b, lfs in curve:
        bval = 'inf' if math.isinf(b) else b
        for s, lf in enumerate(lfs):
            rows.append(dict(device=dev_name, circuit=circ_name,
                             beta=bval, seed=s, lf_cost=lf))
    return dev_name, circ_name, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ibm", action="store_true")
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    args = ap.parse_args()

    tag  = "ibm" if args.ibm else "snail"
    devs = build_ibm_topologies() if args.ibm else build_topology(wraparound=False)

    items = []
    for di, (dev_name, cm, F, basis_gate) in enumerate(devs):
        n_phys = cm.size()
        for name, qc in build_paper_circuits():
            if name in CIRCUITS and qc.num_qubits <= n_phys:
                items.append((di, name))
    print(f"{len(items)} work items, {args.jobs} jobs, tag={tag}")

    os.makedirs("Results", exist_ok=True)
    out_path = f"Results/beta_sweep_{tag}.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["device", "circuit", "beta", "seed", "lf_cost"])
        w.writeheader()
        with ProcessPoolExecutor(max_workers=args.jobs,
                                 initializer=_init, initargs=(tag,)) as pool:
            for dev_name, circ_name, rows in pool.map(_work, items):
                for r in rows:
                    w.writerow(r)
                f.flush()
                best = min(r['lf_cost'] for r in rows)
                print(f"  {dev_name:18s} {circ_name:18s} best lf={best:.3f} "
                      f"({len(rows)} rows)")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
