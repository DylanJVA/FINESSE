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

Usage (beta-sensitivity: 100 warmed layouts per beta, swap_trials=1):
    python3 finesse_beta_sweep.py --seeds 100 --jobs 24         # SNAIL
    python3 finesse_beta_sweep.py --ibm --seeds 100 --jobs 24   # IBM fake_brisbane
"""
import argparse
import csv
import math
import os
import warnings
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

from FrequencyAllocationRuns import (
    build_paper_circuits, build_topology, build_ibm_topologies,
)
from finesse import finesse_transpile

# ~6 circuits spanning sizes and routing demand.
CIRCUITS = ['seca_n11', 'qft_n10', 'multiplier_n15', 'bv_n19', 'qft_n24', 'square_root_n18']
# Sensitivity figure needs a DENSE beta grid (the deployed default is coarse).
DENSE_BETAS = [0, 0.1, 0.2, 0.3, 0.5, 1, 2, 3, 5, 7,
               10, 20, 30, 50, 100, 150, 200, 300, 500, 1000]
# Seeds = warmed layouts. Downstream we take the BEST (min lf) over them at each
# beta, so use ~ the deployed layout budget for min-over-seeds to match deployment.
N_TRAVERSALS = 2
SEED = 0

_DEVS = None
_CIRCS = None
_NSEEDS = 8      # set per-run via --seeds (passed into workers by _init)
_SWAP = 1
_BETAS = DENSE_BETAS   # overridable via --betas (e.g. run only inf), passed by _init


def _init(tag, n_seeds, swap_trials, betas):
    global _DEVS, _CIRCS, _NSEEDS, _SWAP, _BETAS
    _NSEEDS, _SWAP, _BETAS = n_seeds, swap_trials, betas
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
        # swap_trials=1: the curve should show the beta dependence, not the
        # best-of-many routing luck that swap_trials introduces.
        _, _, curve = finesse_transpile(
            qc, cm, F, basis_gate=basis_gate, betas=_BETAS,
            n_seeds=_NSEEDS, n_traversals=N_TRAVERSALS, swap_trials=_SWAP, seed=SEED,
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
    ap.add_argument("--seeds", type=int, default=100,
                    help="warmed layouts per beta; the sensitivity distribution is over these")
    ap.add_argument("--swap-trials", type=int, default=1,
                    help="no effect for beta>0 (fidelity routing rarely ties); keep at 1")
    ap.add_argument("--betas", default=None,
                    help="comma-separated betas to run instead of the dense grid; 'inf' allowed")
    ap.add_argument("--out", default=None,
                    help="output CSV (default Results/beta_sweep_<tag>.csv); use a separate "
                         "file for extra betas, then merge")
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

    betas = (DENSE_BETAS if args.betas is None else
             [float('inf') if b.strip().lower() in ('inf', '∞') else float(b)
              for b in args.betas.split(',')])
    os.makedirs("Results", exist_ok=True)
    out_path = args.out or f"Results/beta_sweep_{tag}.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["device", "circuit", "beta", "seed", "lf_cost"])
        w.writeheader(); f.flush()
        # spawn (not fork): finesse_transpile's rustworkx/rayon thread pool
        # deadlocks in fork()ed workers (same fix as finesse_ablation.py).
        # as_completed: write/print each circuit the moment it finishes, so one
        # slow circuit can't hide progress and the run stays monitorable.
        with ProcessPoolExecutor(max_workers=args.jobs,
                                 mp_context=mp.get_context("spawn"),
                                 initializer=_init,
                                 initargs=(tag, args.seeds, args.swap_trials, betas)) as pool:
            futs = {pool.submit(_work, it): it for it in items}
            for i, fut in enumerate(as_completed(futs), 1):
                dev_name, circ_name, rows = fut.result()
                for r in rows:
                    w.writerow(r)
                f.flush()
                best = min(r['lf_cost'] for r in rows)
                print(f"  [{i}/{len(items)}] {dev_name:18s} {circ_name:18s} "
                      f"best lf={best:.3f} ({len(rows)} rows)", flush=True)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
