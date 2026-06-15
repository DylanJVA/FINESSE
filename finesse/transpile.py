# finesse/transpile.py
#
# User-facing transpilation function.
#
# Strategy: bidir SABRE warmup gives a good initial layout cheaply (emit_ops=False,
# not counted against budget). Then spend the full budget on a flat β grid —
# the high-variance dimension from our sweep analysis.
#
# β grid (21 trials): 20 hand-picked values in [0, 1000] · β=inf (pure fidelity, α=0)
# All trials use the warmed layout. Post-select by lf_cost.
#
# Returns (circuit, best_beta) where best_beta=inf means pure-fidelity won.

from __future__ import annotations

import copy
import math

import numpy as np

from qiskit import QuantumCircuit
from qiskit.converters import dag_to_circuit
from qiskit.transpiler import CouplingMap

from .benchmarks import apply_trivial_layout
from .mirror import circuit_lf_cost
from .routing import route, _layout_pass

# 20 clean β values covering [0, 1000], plus β=inf (pure fidelity) = 21 trials.
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
    seed: int = 0,
) -> tuple[QuantumCircuit, float]:
    """
    Transpile a circuit with FINESSE.

    Runs a flat β grid on a single bidir-warmed initial layout. Post-selects
    the result with the lowest lf_cost.

    β grid: _DEFAULT_BETAS (20 clean values in [0, 1000]) + β=inf (pure fidelity)
    = 21 trials total. Override with the betas argument.

    Args:
        qc:             Circuit to transpile (virtual qubits, unrouted).
        coupling_map:   Device connectivity.
        fidelity_matrix: F[i,j] = 2Q gate fidelity on link (i,j).
        basis_gate:     Native 2Q gate ('sqrt_iswap', 'cx', 'ecr').
        betas:          β values to search. Defaults to _DEFAULT_BETAS.
                        β=inf (pure fidelity) is always appended.
        seed:           RNG seed for the initial random layout.

    Returns:
        (circuit, best_beta): best routed QuantumCircuit and the β that achieved
        it. best_beta=inf means the pure-fidelity trial (α=0) won.
    """
    dag_phys = apply_trivial_layout(qc, coupling_map)
    n_phys = coupling_map.size()
    n_virtual = dag_phys.num_qubits()
    rng = np.random.default_rng(seed)

    # ── Layout: bidir SABRE warmup (emit_ops=False, not counted against budget) ─
    warmup_seed = int(rng.integers(2**31))
    initial_cur = list(rng.permutation(n_phys)[:n_virtual])
    initial_cur = _layout_pass(dag_phys, coupling_map, initial_cur,
                               reverse=False, seed=warmup_seed)
    initial_cur = _layout_pass(dag_phys, coupling_map, initial_cur,
                               reverse=True, seed=warmup_seed)

    # ── β grid ────────────────────────────────────────────────────────────────
    # β=inf sentinel → route with α=0 (pure fidelity), β=1 internally.
    if betas is None:
        betas = _DEFAULT_BETAS
    grid = list(betas) + [float('inf')]

    best_dag, best_cost, best_beta = None, float('inf'), 0.0
    for beta in grid:
        alpha_r = 0.0 if math.isinf(beta) else 1.0
        beta_r  = 1.0 if math.isinf(beta) else beta
        routed, _, _ = route(
            copy.deepcopy(dag_phys), coupling_map,
            seed=warmup_seed, initial_cur=list(initial_cur),
            mode='lightsabre', aggression=2,
            fidelity_matrix=fidelity_matrix,
            fidelity_mirror=True,
            basis_gate=basis_gate,
            alpha=alpha_r, beta=beta_r,
        )
        cost = circuit_lf_cost(routed, fidelity_matrix, basis_gate=basis_gate)
        if cost < best_cost:
            best_dag, best_cost, best_beta = routed, cost, beta

    return dag_to_circuit(best_dag), best_beta
