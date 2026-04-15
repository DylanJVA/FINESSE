# finesse/mirror.py
#
# Mirror gate mechanics: Weyl chamber cost counting and mirror acceptance.
#
# A mirror gate is U' = SWAP @ U. Absorbing a SWAP into an adjacent gate via
# mirroring can reduce the total gate count when decomp_cost(U') < decomp_cost(U) + 3.
#
# Also exports circuit_lf_cost: total −log-fidelity cost of a routed DAG,
# used by both inline_pass.py (trial selection) and ablation.py (benchmarking).

from __future__ import annotations

import warnings

import numpy as np
from qiskit.circuit.library import iSwapGate, CXGate, SwapGate
from qiskit.quantum_info import Operator
from qiskit.synthesis import TwoQubitBasisDecomposer
from qiskit.synthesis.two_qubit.two_qubit_decompose import TwoQubitWeylDecomposition

# ---------------------------------------------------------------------------
# Decomposers — instantiated once at import time.
# Suppress the "not supercontrolled" warning for sqrt_iswap: num_basis_gates()
# is correct for cost counting despite the warning. Actual synthesis uses BQSKit.
# ---------------------------------------------------------------------------
with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    _decomposers: dict[str, TwoQubitBasisDecomposer] = {
        'sqrt_iswap': TwoQubitBasisDecomposer(iSwapGate().power(0.5), euler_basis="ZYZ"),
        'cx':         TwoQubitBasisDecomposer(CXGate(),                euler_basis="ZYZ"),
    }

SUPPORTED_BASIS_GATES = tuple(_decomposers.keys())

# ---------------------------------------------------------------------------
# SWAP unitary (little-endian Qiskit convention)
# ---------------------------------------------------------------------------
SWAP_MATRIX = np.array(
    [[1, 0, 0, 0],
     [0, 0, 1, 0],
     [0, 1, 0, 0],
     [0, 0, 0, 1]],
    dtype=complex,
)


# ---------------------------------------------------------------------------
# Weyl chamber coordinates and decomposition cost
# ---------------------------------------------------------------------------

def weyl_coords(U: np.ndarray) -> tuple[float, float, float]:
    """
    Return the Weyl chamber coordinates (a, b, c) of a 2Q unitary U.
    Satisfies pi/4 >= a >= b >= |c| >= 0.
    Computed via KAK decomposition (TwoQubitWeylDecomposition).
    """
    decomp = TwoQubitWeylDecomposition(U)
    return (decomp.a, decomp.b, decomp.c)


def decomp_cost(U: np.ndarray, basis_gate: str = 'sqrt_iswap') -> int:
    """
    Minimum number of native gates needed to implement 2Q unitary U.
    Returns 0, 1, 2, or 3.

    Uses Qiskit's KAK-based num_basis_gates(), which performs a Weyl chamber
    containment check to determine the exact gate count without synthesizing a circuit.

    Args:
        U:          4×4 unitary matrix.
        basis_gate: Native 2Q gate. Supported: 'sqrt_iswap', 'cx'.
    """
    return _decomposers[basis_gate].num_basis_gates(U)


def mirror_weyl_coords(
    a: float, b: float, c: float
) -> tuple[float, float, float]:
    """
    Equation 1 from the MIRAGE paper: map Weyl coordinates (a, b, c) to the
    Weyl coordinates of the mirror gate U' = SWAP @ U.
    """
    pi4 = np.pi / 4
    if a <= pi4:
        return (pi4 + c, pi4 - b, pi4 - a)
    else:
        return (pi4 - c, pi4 - b, a - pi4)


def circuit_lf_cost(routed_dag, F: np.ndarray,
                    basis_gate: str = 'sqrt_iswap') -> float:
    """
    Total -log-fidelity cost of a routed DAG.
    SWAPs: 3 * (-log F[p0,p1]).  Other 2Q gates: decomp_cost * (-log F[p0,p1]).
    """
    L = -np.log(np.maximum(F, 1e-10))
    total = 0.0
    for node in routed_dag.topological_op_nodes():
        if len(node.qargs) != 2:
            continue
        p0 = routed_dag.find_bit(node.qargs[0]).index
        p1 = routed_dag.find_bit(node.qargs[1]).index
        lf = float(L[p0, p1])
        if isinstance(node.op, SwapGate):
            total += 3.0 * lf
        else:
            try:
                k = float(decomp_cost(Operator(node.op).data, basis_gate))
            except Exception:
                k = 1.0
            total += k * lf
    return total


def accept_mirror(cost_baseline: float, cost_m: float, aggression: int) -> bool:
    """
    Decide whether to accept the mirror form U' = SWAP·U at gate execution time.

    Called from flush_executable() when a 2Q gate is routable. The cost metric
    used for cost_baseline and cost_m is whatever H is configured — pure hop-count
    when no fidelity_matrix is provided, fidelity-weighted when one is:

        No fidelity:      cost = H(layout)          = hop-count sum over stuck F, E
        With fidelity:    cost = k_U * lf + H_fid   where H_fid uses dist_fid paths

    aggression 0: never accept mirror
    aggression 1: accept only if strictly cheaper  (cost_m < cost_baseline)
    aggression 2: accept if cheaper or equal       (cost_m <= cost_baseline)  [default]
    aggression 3: always accept mirror
    """
    if aggression == 0:
        return False
    elif aggression == 1:
        return cost_m < cost_baseline
    elif aggression == 2:
        return cost_m <= cost_baseline
    elif aggression == 3:
        return True
    return False
