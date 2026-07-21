#!/usr/bin/env python3
"""
FINESSE ablation: FASST / MIRAGE / FINESSE, all through finesse_transpile.

All three share the same pipeline (consolidation, bidir-warmed layouts, VF2,
budget, post-select by lf) and differ only by two toggles:

    config    fidelity routing (β search)   mirror
    FASST     yes                            no
    MIRAGE    no                             yes
    FINESSE   yes                            yes   (deployed)

The SABRE baseline is Qiskit's (see qiskit_sabre_baseline.py); it is not run here.

Output: Results/ablation_<tag>.csv with columns
    device, circuit, router, beta, seed, swaps, depth, lf_cost
(beta = NaN for MIRAGE, which has no β).

Usage:
    python3 finesse_ablation.py --jobs 24            # SNAIL
    python3 finesse_ablation.py --ibm --jobs 24      # IBM fake_brisbane
"""
import argparse
import csv
import math
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

# FASST/FINESSE spend the budget on layouts + β (8 seeds × (2+5) = 56 passes).
# MIRAGE has no β, so like SABRE it is a pure layout search: 20 seeds ×
# (2 warmup + 1 route) = 60 passes, matching Qiskit SABRE's 20 trials × FBF.
CONFIGS = {
    'FASST':   dict(aggression=0, use_fidelity=True),
    'MIRAGE':  dict(aggression=2, use_fidelity=False, n_seeds=20),
    'FINESSE': dict(aggression=2, use_fidelity=True),
}
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
    dev_idx, circ_name, cfg_name = item
    dev_name, cm, F, basis_gate = _DEVS[dev_idx]
    qc = _CIRCS[circ_name]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        qc_out, beta = finesse_transpile(
            qc, cm, F, basis_gate=basis_gate,
            seed=SEED, parallel=False, **CONFIGS[cfg_name],
        )
    dag = circuit_to_dag(qc_out)
    return dict(
        device=dev_name, circuit=circ_name, router=cfg_name,
        beta=(np.nan if beta is None else ('inf' if math.isinf(beta) else beta)),
        seed=SEED, swaps=swap_count(dag), depth=dag.depth(),
        lf_cost=circuit_lf_cost(dag, F, basis_gate=basis_gate),
    )


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
            if qc.num_qubits > n_phys:
                continue
            for cfg_name in CONFIGS:
                items.append((di, name, cfg_name))
    print(f"{len(items)} work items, {args.jobs} jobs, tag={tag}")

    os.makedirs("Results", exist_ok=True)
    out_path = f"Results/ablation_{tag}.csv"
    fields = ["device", "circuit", "router", "beta", "seed", "swaps", "depth", "lf_cost"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        with ProcessPoolExecutor(max_workers=args.jobs,
                                 initializer=_init, initargs=(tag,)) as pool:
            for row in pool.map(_work, items):
                w.writerow(row)
                f.flush()
                print(f"  {row['device']:18s} {row['circuit']:18s} "
                      f"{row['router']:8s} beta={row['beta']!s:>5} lf={row['lf_cost']:.3f}")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
