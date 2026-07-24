#!/usr/bin/env python3
"""
Split sweep: compare candidate (seeds x beta x swap_trials) budget splits at a
matched ~520-pass budget, to pick the deployed default. Warmup fixed at 2
(n_traversals=2). Grid vs staged structures both included.

Budget accounting (warmup=2, n_beta = len(betas)+1 because inf is appended):
    grid   : n_seeds * (2 + n_beta * swap_trials)
    staged : n_seeds * (2 + n_beta + swap_trials - 1)

SABRE baseline (opt-3 default, 20/20) is run separately by qiskit_sabre_baseline.py
and joined downstream. This script writes one row per (device, circuit, config)
with the post-selected (min-lf) result.

Usage:
    python3 finesse_split_sweep.py --jobs 24          # SNAIL
    python3 finesse_split_sweep.py --ibm --jobs 24    # IBM fake_brisbane
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

WARMUP = 2   # n_traversals; locked by the warmup study (flat after one forward-back)

# betas are the FINITE points; finesse_transpile appends inf, so n_beta = len+1.
CANDIDATES = {
    #  id       staged  seeds  betas (finite; +inf)                    swap
    'G1':  dict(staged=False, n_seeds=30, betas=[0,1,10,100],                 swap_trials=3),
    'G2':  dict(staged=False, n_seeds=20, betas=[0,10,100],                   swap_trials=6),
    'S1':  dict(staged=True,  n_seeds=30, betas=[0,0.5,2,10,40,150,500],      swap_trials=8),
    'S2':  dict(staged=True,  n_seeds=26, betas=[0,0.3,1,3,10,30,100,300,1000], swap_trials=9),
    'S3':  dict(staged=True,  n_seeds=40, betas=[0,1,10,50,200],              swap_trials=6),
    'S4':  dict(staged=True,  n_seeds=50, betas=[0,1,10,100],                 swap_trials=4),  # push seeds harder
    'NOB': dict(staged=False, n_seeds=20, betas=[],                           swap_trials=24),  # n_beta=1 (inf only)
}
SEED = 0


def budget(cfg):
    n_beta = len(cfg['betas']) + 1
    if cfg['staged']:
        return cfg['n_seeds'] * (WARMUP + n_beta + cfg['swap_trials'] - 1)
    return cfg['n_seeds'] * (WARMUP + n_beta * cfg['swap_trials'])


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
    dev_idx, circ_name, cfg_name = item
    dev_name, cm, F, basis_gate = _DEVS[dev_idx]
    qc = _CIRCS[circ_name]
    cfg = CANDIDATES[cfg_name]
    kw = {k: v for k, v in cfg.items()}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, _, recs = finesse_transpile(
            qc, cm, F, basis_gate=basis_gate, seed=SEED, parallel=False,
            return_all=True, n_traversals=WARMUP, use_vf2=True,
            use_fidelity=True, aggression=2, **kw,
        )
    # Write EVERY trial so post-selection (lf / depth / swaps / gates) is a
    # downstream choice, not baked in here. For grid, one row per (layout, beta);
    # for staged, one row per layout (the scan+refine winner). NOTE: staged
    # already commits to the swap-selection metric (default lf) during its beta
    # scan, so staged rows cannot be faithfully re-post-selected by another
    # metric -- grid preserves that flexibility, staged trades it for budget.
    b = budget(cfg)
    return [dict(
        device=dev_name, circuit=circ_name, config=cfg_name, budget=b,
        seed=r['seed'], beta=('inf' if isinstance(r['beta'], float)
                              and math.isinf(r['beta']) else r['beta']),
        swaps=r['swaps'], depth=r['depth'], gates=r['gates'], lf_cost=r['lf_cost'],
    ) for r in recs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ibm", action="store_true")
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--circuits", type=str, default=None,
                    help="comma-separated circuit names to restrict to (small run)")
    ap.add_argument("--out", type=str, default=None, help="override output path")
    args = ap.parse_args()

    tag = "ibm" if args.ibm else "snail"
    devs = build_ibm_topologies() if args.ibm else build_topology(wraparound=False)
    only = set(args.circuits.split(",")) if args.circuits else None

    print("config budgets:", {k: budget(v) for k, v in CANDIDATES.items()})

    items = []
    for di, (dev_name, cm, F, basis_gate) in enumerate(devs):
        n_phys = cm.size()
        for name, qc in build_paper_circuits():
            if qc.num_qubits > n_phys or (only and name not in only):
                continue
            for cfg_name in CANDIDATES:
                items.append((di, name, cfg_name))
    print(f"{len(items)} work items, {args.jobs} jobs, tag={tag}")

    os.makedirs("Results", exist_ok=True)
    out_path = args.out or f"Results/split_sweep_{tag}.csv"
    fields = ["device", "circuit", "config", "budget",
              "seed", "beta", "swaps", "depth", "gates", "lf_cost"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        n = 0
        with ProcessPoolExecutor(max_workers=args.jobs,
                                 initializer=_init, initargs=(tag,)) as pool:
            for rows in pool.map(_work, items):
                for row in rows:
                    w.writerow(row)
                n += len(rows)
                f.flush()
                r0 = rows[0]
                best = min(r['lf_cost'] for r in rows)
                print(f"  {r0['device']:16s} {r0['circuit']:16s} "
                      f"{r0['config']:4s} {len(rows):3d} trials  best_lf={best:.4f}")
    print(f"\nWrote {out_path} ({n} rows)")


if __name__ == "__main__":
    main()
