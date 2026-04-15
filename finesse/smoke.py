# finesse/smoke.py
#
# Smoke-check assertions for fidelity-aware routing invariants.
#
# Key invariant
# -------------
# When the fidelity matrix F is uniform (all coupling edges have the same value),
# D_fid is proportional to D_hop (all edge weights equal), so relative distances
# are preserved. Therefore routing with uniform F must produce the same SWAP
# count as routing without fidelity_matrix (pure hop-count).
#
# run_smoke_checks() exercises this and checks that heterogeneous F causes
# fidelity-aware routing to diverge from hop-count routing.
# Call it from run_ablation.py, a notebook, or a CI job.

from __future__ import annotations

import copy
import warnings

import numpy as np

from .ablation import make_uniform_fidelity, make_synthetic_fidelity
from .benchmarks import apply_trivial_layout, fetch_qasm
from .routing import route


def _swap_count(dag) -> int:
    from qiskit.converters import dag_to_circuit
    return dag_to_circuit(dag).count_ops().get('swap', 0)


def check_uniform_fidelity_invariant(
    qasm_name: str,
    coupling_map,
    *,
    n_seeds: int = 3,
    verbose: bool = True,
) -> bool:
    """
    Assert: routing with uniform F gives the same SWAP count as
    routing without fidelity_matrix (pure hop-count), for every seed.

    Invariant: with uniform F, D_fid ∝ D_hop (all edges equal), so SWAP
    rankings are preserved.

    Returns True if all seeds pass, False (with a printed report) otherwise.
    """
    F_uniform = make_uniform_fidelity(coupling_map)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        qc = fetch_qasm(qasm_name)
        dag_phys = apply_trivial_layout(qc.copy(), coupling_map)

    all_ok = True
    for seed in range(n_seeds):
        # Baseline: MIRAGE, no fidelity_matrix (pure hop-count H)
        rd_base, _, _ = route(
            copy.deepcopy(dag_phys), coupling_map,
            seed=seed, mode='lightsabre', aggression=2,
        )
        sw_base = _swap_count(rd_base)

        rd_fid, _, _ = route(
            copy.deepcopy(dag_phys), coupling_map,
            seed=seed, mode='lightsabre', aggression=2,
            fidelity_matrix=F_uniform,
        )
        sw_fid = _swap_count(rd_fid)
        ok = (sw_fid == sw_base)

        if verbose or not ok:
            status = 'OK' if ok else 'FAIL'
            print(
                f"  [{status}] seed={seed}  "
                f"base={sw_base}  "
                f"uniform(fid)={sw_fid} {'==' if ok else '!='} base"
            )
        all_ok = all_ok and ok

    return all_ok


def check_heterogeneous_fidelity_sensitivity(
    qasm_name: str,
    coupling_map,
    *,
    n_seeds: int = 5,
    verbose: bool = True,
) -> bool:
    """
    Sanity-check: with heterogeneous (synthetic) F, fidelity-aware routing
    (fidelity_matrix provided) should produce different SWAP counts from
    pure hop-count routing on at least one seed.

    This is advisory — it cannot be deterministically true for every circuit
    and seed.  Returns True if any seed shows a difference.
    """
    F_synth = make_synthetic_fidelity(coupling_map, seed=42)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        qc = fetch_qasm(qasm_name)
        dag_phys = apply_trivial_layout(qc.copy(), coupling_map)

    diffs = []
    for seed in range(n_seeds):
        rd_base, _, _ = route(
            copy.deepcopy(dag_phys), coupling_map,
            seed=seed, mode='lightsabre', aggression=2,
        )
        rd_fid, _, _ = route(
            copy.deepcopy(dag_phys), coupling_map,
            seed=seed, mode='lightsabre', aggression=2,
            fidelity_matrix=F_synth,
        )
        sw_base = _swap_count(rd_base)
        sw_fid  = _swap_count(rd_fid)
        diff = sw_fid - sw_base
        diffs.append(diff)
        if verbose:
            marker = '←' if diff != 0 else ''
            print(f"  seed={seed}  base: {sw_base} swaps  fid: {sw_fid} swaps  Δ={diff:+d} {marker}")

    any_diff = any(d != 0 for d in diffs)
    if verbose:
        if any_diff:
            print(f"  → fidelity-aware routing differs from hop-count on heterogeneous F (expected)")
        else:
            print(f"  → WARNING: fidelity_matrix never changed SWAP counts — may indicate no signal")
    return any_diff


def run_smoke_checks(
    coupling_map=None,
    circuits: list[str] | None = None,
    n_seeds: int = 3,
) -> bool:
    """
    Run both smoke checks:
      1. Uniform-F invariant: route(uniform F) == route(no fidelity)
      2. Heterogeneous-F sensitivity: fidelity routing differs from hop-count on real F.

    Args:
        coupling_map: Defaults to Q20 Tokyo if not provided.
        circuits:     List of QASM filenames (default: a small subset of the suite).
        n_seeds:      Seeds per circuit.
    Returns True if all invariant checks pass (sensitivity is advisory only).
    """
    if coupling_map is None:
        from .benchmarks import make_tokyo
        coupling_map = make_tokyo()
    if circuits is None:
        circuits = ['4gt11_84.qasm', 'rd32-v0_66.qasm']

    print("Smoke check 1/2: uniform-F invariant")
    print("  route(uniform F) must equal route(no fidelity_matrix)")
    print("  because D_fid ∝ D_hop when F is uniform")
    print()

    all_ok = True
    for circ in circuits:
        print(f"  [{circ}]")
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            ok = check_uniform_fidelity_invariant(
                circ, coupling_map, n_seeds=n_seeds,
            )
        result = 'PASS' if ok else 'FAIL'
        print(f"    → {result}\n")
        all_ok = all_ok and ok

    print("Smoke check 2/2: heterogeneous-F sensitivity")
    print("  fidelity routing should differ from hop-count on real F (advisory)")
    print()
    any_sensitive = False
    for circ in circuits:
        print(f"  [{circ}]")
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            sensitive = check_heterogeneous_fidelity_sensitivity(
                circ, coupling_map, n_seeds=n_seeds,
            )
        any_sensitive = any_sensitive or sensitive
        print()

    print('=' * 50)
    print(f"Invariant (uniform F): {'ALL PASS' if all_ok else 'FAILURES'}")
    print(f"Sensitivity (real F):  {'signal detected' if any_sensitive else 'WARNING: no signal'}")
    return all_ok


if __name__ == '__main__':
    run_smoke_checks()
