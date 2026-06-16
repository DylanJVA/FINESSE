# finesse/transpile.py
#
# Strategy: run a (seed × β) grid. Each seed gets its own bidir SABRE warmup
# to find a good initial layout, then is routed with every β value. All trials
# are independent and run in parallel. Post-select by lf_cost.
#
# Default: 3 seeds × 21 β values = 63 trials, all parallelised over available cores.

from __future__ import annotations

import copy
import math
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from qiskit import QuantumCircuit
from qiskit.converters import dag_to_circuit
from qiskit.transpiler import CouplingMap

from .benchmarks import apply_trivial_layout
from .mirror import circuit_lf_cost
from .routing import route, _layout_pass

_DEFAULT_BETAS = [
    0, 0.1, 0.2, 0.3, 0.5, 1, 2, 3, 5, 7,
    10, 20, 30, 50, 100, 150, 200, 300, 500, 1000,
]


def finesse_transpile(
    qc: QuantumCircuit,
    coupling_map: CouplingMap,
    fidelity_matrix: np.ndarray,
    basis_gate: str = 'sqrt_iswap',
    betas: list[float] | None = None,
    n_seeds: int = 3,
    seed: int = 0,
) -> tuple[QuantumCircuit, float]:
    """
    Transpile a circuit with FINESSE.

    Runs a (seed × β) grid. Each seed generates a distinct bidir-warmed layout;
    every β is tried with every layout. All trials run in parallel via
    ThreadPoolExecutor (LightSABRE is Rust-based and releases the GIL).
    Post-selects the result with the lowest lf_cost.

    Args:
        qc:             Circuit to transpile (virtual qubits, unrouted).
        coupling_map:   Device connectivity.
        fidelity_matrix: F[i,j] = 2Q gate fidelity on link (i,j).
        basis_gate:     Native 2Q gate ('sqrt_iswap', 'cx', 'ecr').
        betas:          β values to search. β=inf (pure fidelity) always appended.
        n_seeds:        Number of distinct bidir-warmed layouts to try.
        seed:           Master RNG seed.

    Returns:
        (circuit, best_beta): best routed QuantumCircuit and the β that achieved it.
    """
    dag_phys = apply_trivial_layout(qc, coupling_map)
    n_phys   = coupling_map.size()
    n_virtual = dag_phys.num_qubits()
    rng = np.random.default_rng(seed)

    if betas is None:
        betas = _DEFAULT_BETAS
    grid_betas = list(betas) + [float('inf')]

    # ── Generate n_seeds bidir-warmed layouts ─────────────────────────────────
    layouts = []
    for _ in range(n_seeds):
        warmup_seed = int(rng.integers(2**31))
        initial = list(rng.permutation(n_phys)[:n_virtual])
        initial = _layout_pass(dag_phys, coupling_map, initial,
                               reverse=False, seed=warmup_seed)
        initial = _layout_pass(dag_phys, coupling_map, initial,
                               reverse=True,  seed=warmup_seed)
        layouts.append((warmup_seed, initial))

    # ── (layout, β) grid — all independent, run in parallel ──────────────────
    trials = [(ws, cur, beta) for ws, cur in layouts for beta in grid_betas]

    def _trial(args):
        warmup_seed, initial_cur, beta = args
        alpha_r = 0.0 if math.isinf(beta) else 1.0
        beta_r  = 1.0 if math.isinf(beta) else beta
        F = fidelity_matrix.copy()
        routed, _, _ = route(
            copy.deepcopy(dag_phys), coupling_map,
            seed=warmup_seed, initial_cur=list(initial_cur),
            mode='lightsabre', aggression=2,
            fidelity_matrix=F,
            fidelity_mirror=True,
            basis_gate=basis_gate,
            alpha=alpha_r, beta=beta_r,
        )
        cost = circuit_lf_cost(routed, F, basis_gate=basis_gate)
        return routed, cost, beta

    with ThreadPoolExecutor(max_workers=len(trials)) as pool:
        results = list(pool.map(_trial, trials))

    best_dag, best_cost, best_beta = None, float('inf'), 0.0
    for routed, cost, beta in results:
        if cost < best_cost:
            best_dag, best_cost, best_beta = routed, cost, beta

    return dag_to_circuit(best_dag), best_beta
