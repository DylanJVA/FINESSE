#!/usr/bin/env python3
"""
Sweep FINESSE's layout-warmup depth (n_traversals) and compare to Qiskit's
shipped SABRE.

For each (device, circuit) runs finesse_transpile with n_traversals in
{0,1,2,3} (0 = no warmup / random layout, 2 = forward-backward, 3 =
forward-backward-forward). Each setting still does the full n_seeds × β grid.

Parallelised across (device, circuit, n_traversals) work items with a process
pool (--jobs); each worker runs its β grid serially to avoid oversubscription.

Output: Results/traversals_<tag>.csv with schema
    device, circuit, router, beta, seed, swaps, depth, lf_cost
where router = 'FINESSE_t{n}'. Drops into the existing plotting code.

Usage:
    python3 finesse_traversals_sweep.py --jobs 24                 # SNAIL, small+med
    python3 finesse_traversals_sweep.py --ibm --jobs 24
"""
import argparse
import csv
import os
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from qiskit.converters import circuit_to_dag

from FrequencyAllocationRuns import (
    build_paper_circuits, build_topology, build_ibm_topologies,
)
from finesse import finesse_transpile
from finesse.benchmarks import swap_count
from finesse.mirror import circuit_lf_cost

N_TRAVERSALS = [0, 1, 2, 3]
N_SEEDS = 3
SEED = 0

_DEVS = None
_CIRCS = None


def _init(tag):
    """Per-worker: build devices and circuits once."""
    global _DEVS, _CIRCS
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _DEVS = (build_ibm_topologies() if tag == "ibm"
                 else build_topology(wraparound=False))
        _CIRCS = {n: qc for n, qc in build_paper_circuits()}


def _work(item):
    """Run one work item: (device idx, circuit, n_traversals, use_vf2, router)."""
    dev_idx, circ_name, nt, use_vf2, router = item
    dev_name, cm, F, basis_gate = _DEVS[dev_idx]
    qc = _CIRCS[circ_name]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        qc_out, beta = finesse_transpile(
            qc, cm, F, basis_gate=basis_gate,
            n_seeds=N_SEEDS, n_traversals=nt, seed=SEED,
            parallel=False, use_vf2=use_vf2,
        )
    dag = circuit_to_dag(qc_out)
    lf  = circuit_lf_cost(dag, F, basis_gate=basis_gate)
    return dict(device=dev_name, circuit=circ_name, router=router,
                beta=beta, seed=SEED,
                swaps=swap_count(dag), depth=dag.depth(), lf_cost=lf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ibm", action="store_true")
    ap.add_argument("--max-qubits", type=int, default=20,
                    help="skip circuits larger than this (quick results)")
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    args = ap.parse_args()

    tag  = "ibm" if args.ibm else "snail"
    devs = build_ibm_topologies() if args.ibm else build_topology(wraparound=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        circuits = build_paper_circuits()

    # Build the work list. Traversal study (VF2 off) + one VF2-on variant (t=2).
    items = []
    for di, (dev_name, cm, F, basis_gate) in enumerate(devs):
        n_phys = cm.size()
        for name, qc in circuits:
            if qc.num_qubits > n_phys or qc.num_qubits > args.max_qubits:
                continue
            for nt in N_TRAVERSALS:
                items.append((di, name, nt, False, f"FINESSE_t{nt}"))
                items.append((di, name, nt, True,  f"FINESSE_t{nt}_vf2"))
    print(f"{len(items)} work items, {args.jobs} jobs, tag={tag}")

    os.makedirs("Results", exist_ok=True)
    out_path = f"Results/traversals_{tag}.csv"
    fieldnames = ["device", "circuit", "router", "beta", "seed",
                  "swaps", "depth", "lf_cost"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        with ProcessPoolExecutor(max_workers=args.jobs,
                                 initializer=_init, initargs=(tag,)) as pool:
            for row in pool.map(_work, items):
                w.writerow(row)
                f.flush()
                print(f"  {row['device']:18s} {row['circuit']:18s} "
                      f"{row['router']:11s} beta={row['beta']!s:>5} "
                      f"lf={row['lf_cost']:.3f}")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
