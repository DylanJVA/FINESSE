#!/usr/bin/env python3
"""
Qiskit shipped-SABRE baseline.

Runs Qiskit's native SabreLayout (Rust, forward-backward-forward via
max_iterations=3, with internal layout_trials/swap_trials post-selection) on the
SAME consolidated 2Q-block circuits FINESSE routes. One call per (device,
circuit) — Qiskit does its own multi-trial post-selection internally.

Output: Results/qiskit_sabre_<tag>.csv with the same schema as transpile_*.csv
(device, circuit, router, beta, seed, swaps, depth, lf_cost) so it drops straight
into the existing plotting code. router='QISKIT_SABRE', beta=NaN.

Usage:
    python3 qiskit_sabre_baseline.py            # SNAIL topologies
    python3 qiskit_sabre_baseline.py --ibm      # IBM fake_brisbane
"""
import argparse
import csv
import os
import warnings

import numpy as np
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import SabreLayout
from qiskit.converters import circuit_to_dag

from FrequencyAllocationRuns import (
    build_paper_circuits, build_topology, build_ibm_topologies,
)
from finesse.benchmarks import make_unroll_consolidate
from finesse.mirror import circuit_lf_cost, circuit_2q_gate_count

# Match LightSABRE's default (LD) config: 20 layout trials, 20 swap trials.
LAYOUT_TRIALS = 20
SWAP_TRIALS   = 20
SEED          = 0


def run_qiskit_sabre(qc_cons, cm, F, basis_gate):
    """One Qiskit SabreLayout call (layout + routing, internal post-selection)."""
    pm = PassManager([
        SabreLayout(cm, seed=SEED,
                    swap_trials=SWAP_TRIALS, layout_trials=LAYOUT_TRIALS),
    ])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        routed_qc = pm.run(qc_cons)
    dag = circuit_to_dag(routed_qc)
    swaps = sum(1 for inst in routed_qc.data if inst.operation.name == 'swap')
    return dict(
        swaps=swaps,
        depth=dag.depth(),
        gates=circuit_2q_gate_count(dag, basis_gate=basis_gate),
        lf_cost=circuit_lf_cost(dag, F, basis_gate=basis_gate),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ibm", action="store_true", help="run on IBM fake_brisbane")
    args = ap.parse_args()

    tag  = "ibm" if args.ibm else "snail"
    devs = build_ibm_topologies() if args.ibm else build_topology(wraparound=False)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        circuits = build_paper_circuits()

    # Consolidate each circuit once (device-independent prep).
    cons = {}
    for name, qc in circuits:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cons[name] = PassManager(make_unroll_consolidate()).run(qc)

    os.makedirs("Results", exist_ok=True)
    out_path = f"Results/qiskit_sabre_{tag}.csv"
    fieldnames = ["device", "circuit", "router", "beta", "seed",
                  "swaps", "depth", "gates", "lf_cost"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for dev_name, cm, F, basis_gate in devs:
            n_phys = cm.size()
            for name, _ in circuits:
                qc_cons = cons[name]
                if qc_cons.num_qubits > n_phys:
                    print(f"  skip {dev_name}/{name} ({qc_cons.num_qubits}q > {n_phys}q)")
                    continue
                res = run_qiskit_sabre(qc_cons, cm, F, basis_gate)
                row = dict(device=dev_name, circuit=name,
                           router="QISKIT_SABRE", beta=np.nan, seed=SEED, **res)
                w.writerow(row)
                f.flush()
                print(f"  {dev_name:18s} {name:18s} "
                      f"swaps={res['swaps']:4d} depth={res['depth']:4d} "
                      f"lf={res['lf_cost']:.3f}")

    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
