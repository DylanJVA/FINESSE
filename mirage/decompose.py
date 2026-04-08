# mirage/decompose.py

from __future__ import annotations

import cirq
import numpy as np

from qiskit.dagcircuit import DAGCircuit
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate, iSwapGate, SwapGate
from qiskit.quantum_info import Operator
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.converters import circuit_to_dag
from qiskit.synthesis import OneQubitEulerDecomposer

_1q_decomposer = OneQubitEulerDecomposer(basis="ZYZ")

# SWAP matrix for Cirq→Qiskit endianness correction.
# Cirq is big-endian (qubit 0 = MSB), Qiskit is little-endian (qubit 0 = LSB).
# Conjugating a 2Q unitary by SWAP_MATRIX fixes the qubit ordering.
_SWAP_MATRIX = np.array(
    [[1, 0, 0, 0],
     [0, 0, 1, 0],
     [0, 1, 0, 0],
     [0, 0, 0, 1]],
    dtype=complex,
)


def _unitary_to_sqrt_iswap_circuit(U: np.ndarray) -> QuantumCircuit:
    """
    Decompose a 4x4 unitary into sqrt(iSWAP) + 1Q gates using Cirq's
    exact decomposer, then convert the result back to Qiskit.

    Qiskit's synthesis methods assume a supercontrolled basis gate, which
    sqrt(iSWAP) is not. Cirq's two_qubit_matrix_to_sqrt_iswap_operations
    handles sqrt(iSWAP) correctly.

    Endianness fix: Cirq and Qiskit index tensor products in opposite order.
    - 1Q gates: reverse qubit index (Cirq qubit i → Qiskit qubit n-1-i)
    - 2Q gates: additionally conjugate unitary by SWAP_MATRIX
    """
    q0, q1 = cirq.LineQubit.range(2)
    ops = cirq.two_qubit_matrix_to_sqrt_iswap_operations(q0, q1, U)
    cirq_circuit = cirq.Circuit(ops)

    qubits = sorted(cirq_circuit.all_qubits())
    if not qubits:
        # Cirq returned empty circuit (near-identity unitary)
        return QuantumCircuit(2)

    n = len(qubits)
    qubit_index = {q: (n - 1 - i) for i, q in enumerate(qubits)}
    qc = QuantumCircuit(n)

    sqrt_iswap = iSwapGate().power(0.5)

    for op in cirq_circuit.all_operations():
        gate = op.gate
        qargs = [qubit_index[q] for q in op.qubits]
        U_op = cirq.unitary(op)

        if isinstance(gate, cirq.ISwapPowGate) and gate.exponent == 0.5:
            # Keep as named gate for gate counting and identification
            qc.append(sqrt_iswap, qargs)
        elif len(op.qubits) == 1:
            # Decompose 1Q unitary into ZYZ Euler angles
            qc.compose(_1q_decomposer(U_op), qubits=qargs, inplace=True)
        else:
            # Other 2Q gates: conjugate by SWAP for endianness
            qc.append(UnitaryGate(_SWAP_MATRIX @ U_op @ _SWAP_MATRIX,
                                  check_input=False), qargs)

    return qc


class MirageDecompose(TransformationPass):
    """
    Decompose all 2Q gates into sqrt(iSWAP) + ZYZ single-qubit gates.

    Handles UnitaryGate and SwapGate nodes (and any other 2Q gate whose
    unitary can be obtained via Operator). Gates already in the
    sqrt(iSWAP) basis (name "xx_plus_yy") are left unchanged.

    Uses Cirq's exact sqrt(iSWAP) decomposer for correctness.
    """

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        nodes_to_decompose = [
            node for node in dag.op_nodes()
            if len(node.qargs) == 2
            and node.op.name != "xx_plus_yy"  # already in basis
        ]

        for node in nodes_to_decompose:
            try:
                U = Operator(node.op).data
            except Exception:
                continue

            decomposed_circuit = _unitary_to_sqrt_iswap_circuit(U)

            if decomposed_circuit.num_qubits < 2:
                # Cirq returned empty — skip (near-identity, no 2Q content)
                continue

            decomposed_dag = circuit_to_dag(decomposed_circuit)
            wire_map = {
                decomposed_circuit.qubits[0]: node.qargs[0],
                decomposed_circuit.qubits[1]: node.qargs[1],
            }
            dag.substitute_node_with_dag(node, decomposed_dag, wires=wire_map)

        return dag
