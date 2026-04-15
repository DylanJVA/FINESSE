# finesse/fidelity.py
#
# Utilities for building fidelity matrices from backend calibration data
# and constructing Qiskit Targets that encode per-link error rates.

from __future__ import annotations

import numpy as np
from qiskit.circuit.library import iSwapGate, CXGate, UGate
from qiskit.transpiler import CouplingMap, Target, InstructionProperties


def build_target_from_fidelities(
    coupling_map: CouplingMap,
    fidelity_matrix: np.ndarray,
) -> Target:
    """
    Build a Qiskit Target encoding gate error rates from a fidelity matrix.

    Populates sqrt(iSWAP), CX, and U gate error rates so that layout passes
    (e.g. VF2Layout) can use them for fidelity-aware qubit placement.

    Note: SabreSwap does NOT use these error rates for SWAP selection — its
    heuristic is purely distance-based. For fidelity-aware routing, use
    InlineMirageSwap with fidelity_matrix directly.

    Args:
        coupling_map:    Device connectivity.
        fidelity_matrix: F[i,j] = fidelity of sqrt(iSWAP) on link (i,j).

    Gate error rates:
        sqrt(iSWAP): error = 1 - F[i,j]
        CX:          error = 1 - F[i,j]^2   (CX costs 2 sqrt(iSWAP) gates)
        U:           error = 0.0             (single-qubit gates assumed perfect)
    """
    target = Target(num_qubits=coupling_map.size())
    sqrt_iswap = iSwapGate().power(0.5)

    sqiswap_props = {}
    cx_props = {}
    for p0, p1 in coupling_map.get_edges():
        f = max(float(fidelity_matrix[p0, p1]), 1e-10)
        sqiswap_props[(p0, p1)] = InstructionProperties(error=1.0 - f)
        cx_props[(p0, p1)]      = InstructionProperties(error=1.0 - f ** 2)

    target.add_instruction(sqrt_iswap, sqiswap_props)
    target.add_instruction(CXGate(), cx_props)

    u_props = {
        (p,): InstructionProperties(error=0.0)
        for p in range(coupling_map.size())
    }
    target.add_instruction(UGate(0, 0, 0), u_props)

    return target


def fidelity_matrix_from_backend(backend) -> tuple[CouplingMap, np.ndarray, list[int]]:
    """
    Extract per-link sqrt(iSWAP) fidelities from a backend's calibration data.

    Supports backends whose native 2Q gate is CX or ECR. Since CX decomposes
    into k=2 sqrt(iSWAP) gates, the per-gate sqrt(iSWAP) fidelity is:

        F_sqiswap[i,j] = sqrt(F_cx[i,j]) = sqrt(1 - e_cx[i,j])

    The same formula applies to ECR since it is locally equivalent to CX.

    Returns:
        coupling_map: CouplingMap built from links with valid calibration data.
        F:            np.ndarray of shape (n, n) where F[i,j] is the
                      sqrt(iSWAP) fidelity on logical link (i,j).
        physical_qubits: list mapping logical index i -> physical qubit index.
    """
    target = backend.target

    # Find the native 2Q gate
    two_q_gate = None
    for name in ("cx", "ecr", "cz"):
        if name in target.operation_names:
            two_q_gate = name
            break
    if two_q_gate is None:
        raise ValueError(
            f"Backend {backend.name} has no recognised 2Q gate "
            f"(cx, ecr, cz). Found: {target.operation_names}"
        )

    # Extract valid per-link error rates
    link_errors = {}
    for qargs, props in target[two_q_gate].items():
        if (qargs is not None and len(qargs) == 2
                and props is not None
                and props.error is not None
                and 0 < props.error < 0.5):
            link_errors[qargs] = props.error

    if not link_errors:
        raise ValueError(f"No valid calibration data found for {two_q_gate} on {backend.name}")

    # BFS over calibrated edges to find the largest connected subgraph
    calibrated_neighbors: dict[int, list[int]] = {}
    for (p0, p1) in link_errors:
        calibrated_neighbors.setdefault(p0, []).append(p1)
        calibrated_neighbors.setdefault(p1, []).append(p0)

    best_root = max(calibrated_neighbors, key=lambda q: len(calibrated_neighbors[q]))
    visited: list[int] = []
    queue = [best_root]
    seen = {best_root}
    while queue:
        q = queue.pop(0)
        visited.append(q)
        for nb in calibrated_neighbors.get(q, []):
            if nb not in seen:
                seen.add(nb)
                queue.append(nb)

    selected = set(visited)
    phys_to_idx = {phys: idx for idx, phys in enumerate(visited)}
    n = len(visited)

    # Build coupling map from calibrated edges within the subgraph
    subgraph_edges = [
        (phys_to_idx[p0], phys_to_idx[p1])
        for (p0, p1) in link_errors
        if p0 in selected and p1 in selected
    ]
    coupling_map = CouplingMap(couplinglist=subgraph_edges)

    # Build F matrix: F_sqiswap = sqrt(1 - native_error)
    F = np.eye(n)
    for (p0, p1), e in link_errors.items():
        if p0 in phys_to_idx and p1 in phys_to_idx:
            i = phys_to_idx[p0]
            j = phys_to_idx[p1]
            f = np.sqrt(1.0 - e)
            F[i, j] = f
            F[j, i] = f

    return coupling_map, F, visited