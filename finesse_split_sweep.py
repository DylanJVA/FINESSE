#!/usr/bin/env python3
"""
Split sweep: compare candidate (seeds x beta x swap_trials) budget splits at a
budget matched to Qiskit SABRE (~600 passes: 24 layout trials x (6 warmup + 20
swap)), to pick the deployed default. Warmup fixed at 6 (n_traversals=6, i.e.
SABRE's 2*max_iterations=6); routing-dominated circuits (e.g. bv_n19) need the
deeper warmup to reach a good layout, and the earlier warmup=2 study omitted
them. Grid vs staged structures both included.

Budget accounting (warmup=6; total_layouts = n_seeds random + 3 heuristic
[dense/identity/reversed, like SABRE]; n_beta = len(betas)+1 as inf is appended):
    grid   : total_layouts * (6 + n_beta * swap_trials)
    staged : total_layouts * (6 + n_beta + swap_trials - 1)

SABRE baseline (opt-3 default, 20/20) is run separately by qiskit_sabre_baseline.py
and joined downstream. This script writes EVERY trial (one row per (layout, beta)
for grid configs, one per layout for staged), each row self-documenting the full
configuration, so post-selection (lf / depth / swaps) is a downstream choice.

Usage:
    python3 finesse_split_sweep.py --jobs 24          # SNAIL
    python3 finesse_split_sweep.py --ibm --jobs 24    # IBM fake_brisbane
"""
import argparse
import csv
import math
import multiprocessing as mp
import os
import warnings
from concurrent.futures import ProcessPoolExecutor

from FrequencyAllocationRuns import (
    build_paper_circuits, build_topology, build_ibm_topologies,
)
from finesse import finesse_transpile

WARMUP = 6      # n_traversals = 2*max_iterations, matching SABRE; deeper warmup
                # fixes routing-dominated circuits (bv_n19: warmup 2->6 turns a
                # +16% loss into a -13% win) that the warmup=2 study missed.
N_HEURISTIC = 3 # dense/identity/reversed layouts always added (like SABRE's), so
                # a config's total warmed layouts = n_seeds (random) + N_HEURISTIC.

# n_seeds is the RANDOM-seed count; 3 heuristic layouts are added on top, so total
# layouts = n_seeds + 3 and budget uses that total. betas are FINITE points
# (finesse_transpile appends inf, so n_beta = len+1). Tuned to ~600 (SABRE's real
# budget: 598 SNAIL / 624 IBM).
CANDIDATES = {
    #  id       staged  random  betas (finite; +inf)              swap    total  budget
    'S4':  dict(staged=True,  n_seeds=47, betas=[0,1,10,100],         swap_trials=2),  # 50  600  seed-heavy staged
    'S3':  dict(staged=True,  n_seeds=37, betas=[0,1,10,50,200],      swap_trials=4),  # 40  600  staged runner-up
    'G1':  dict(staged=False, n_seeds=27, betas=[0,10,100],           swap_trials=3),  # 30  540  seed-heavy grid
    'G2':  dict(staged=False, n_seeds=17, betas=[0,10,100],           swap_trials=6),  # 20  600  swap-heavy grid
}
SEED = 0


def budget(cfg):
    n_beta = len(cfg['betas']) + 1
    total = cfg['n_seeds'] + N_HEURISTIC          # random seeds + heuristic layouts
    if cfg['staged']:
        return total * (WARMUP + n_beta + cfg['swap_trials'] - 1)
    return total * (WARMUP + n_beta * cfg['swap_trials'])


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
    cfg = CANDIDATES[cfg_name]   # keys: staged, n_seeds, betas, swap_trials
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, _, recs = finesse_transpile(
            qc, cm, F, basis_gate=basis_gate, seed=SEED, parallel=False,
            return_all=True, n_traversals=WARMUP, use_vf2=True,
            use_fidelity=True, aggression=2, **cfg,
        )
    # Write EVERY trial so post-selection (lf / depth / swaps / gates) is a
    # downstream choice, not baked in here. For grid, one row per (layout, beta);
    # for staged, one row per layout (the scan+refine winner). NOTE: staged
    # already commits to the swap-selection metric (default lf) during its beta
    # scan, so staged rows cannot be faithfully re-post-selected by another
    # metric -- grid preserves that flexibility, staged trades it for budget.
    # Every row is self-documenting: it carries the full configuration that
    # produced it, not just the config id, so the CSV stands on its own even if
    # CANDIDATES later changes.
    common = dict(
        device=dev_name, circuit=circ_name, config=cfg_name, budget=budget(cfg),
        structure=('staged' if cfg['staged'] else 'grid'),
        n_seeds=cfg['n_seeds'], n_heuristic=N_HEURISTIC, n_beta=len(cfg['betas']) + 1,
        swap_trials=cfg['swap_trials'], warmup=WARMUP,
    )
    return [dict(
        **common,
        seed=r['seed'], beta=('inf' if isinstance(r['beta'], float)
                              and math.isinf(r['beta']) else r['beta']),
        swaps=r['swaps'], depth=r['depth'], gates=r['gates'], lf_cost=r['lf_cost'],
    ) for r in recs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ibm", action="store_true")
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--circuits", type=str, default=None,
                    help="comma-separated circuit names to restrict to (small validation run)")
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
              "structure", "n_seeds", "n_heuristic", "n_beta", "swap_trials", "warmup",
              "seed", "beta", "swaps", "depth", "gates", "lf_cost"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        n = 0
        # 'spawn' (not the Linux default 'fork'): DenseLayout/VF2Layout use
        # rustworkx's rayon thread pool, which deadlocks in fork()ed children if
        # the parent already initialized it (build_topology does). spawn gives
        # each worker a clean interpreter, avoiding the hang.
        with ProcessPoolExecutor(max_workers=args.jobs,
                                 mp_context=mp.get_context("spawn"),
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
