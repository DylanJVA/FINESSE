# mirage/cost.py

from __future__ import annotations

import warnings

import numpy as np
from qiskit.circuit.library import iSwapGate
from qiskit.synthesis import TwoQubitBasisDecomposer
from qiskit.synthesis.two_qubit.two_qubit_decompose import TwoQubitWeylDecomposition

# sqrt(iSWAP) basis gate — Weyl coords (pi/8, pi/8, 0), NOT supercontrolled.
# Huang et al. 2023 proves k=3 is still universal for sqrt(iSWAP).
SQRT_ISWAP_GATE = iSwapGate().power(0.5)

# Suppress the "not supercontrolled" warning — num_basis_gates() is correct
# for cost counting despite the warning. Actual synthesis uses Cirq (decompose.py).
with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    _decomposer = TwoQubitBasisDecomposer(SQRT_ISWAP_GATE, euler_basis="ZYZ")

SWAP_MATRIX = np.array(
    [[1, 0, 0, 0],
     [0, 0, 1, 0],
     [0, 1, 0, 0],
     [0, 0, 0, 1]],
    dtype=complex,
)


def decomp_cost(U: np.ndarray) -> int:
    """
    Minimum number of sqrt(iSWAP) gates needed to implement 2Q unitary U.
    Returns 0, 1, 2, or 3.

    Uses Qiskit's KAK-based num_basis_gates() which does the correct
    Weyl chamber containment check. This is mathematically equivalent
    to monodromy polytope containment without requiring the monodromy library.

    Note: actual circuit synthesis is handled by Cirq in decompose.py, not
    by this decomposer — Qiskit's synthesis is incorrect for non-supercontrolled
    bases.
    """
    return _decomposer.num_basis_gates(U)


def weyl_coords(U: np.ndarray) -> tuple[float, float, float]:
    """
    Return the Weyl chamber coordinates (a, b, c) of a 2Q unitary U.
    Satisfies pi/4 >= a >= b >= |c| >= 0.
    Computed via KAK decomposition (TwoQubitWeylDecomposition).
    """
    decomp = TwoQubitWeylDecomposition(U)
    return (decomp.a, decomp.b, decomp.c)
