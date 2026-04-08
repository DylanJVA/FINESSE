# mirage/mirror.py

from __future__ import annotations

import numpy as np
from qiskit.dagcircuit import DAGCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Operator

from .cost import decomp_cost, SWAP_MATRIX


def mirror_weyl_coords(
    a: float, b: float, c: float
) -> tuple[float, float, float]:
    """
    Equation 1 from the paper: map Weyl coordinates (a, b, c) to the
    Weyl coordinates of the mirror gate U' = SWAP @ U.
    """
    pi4 = np.pi / 4
    if a <= pi4:
        return (pi4 + c, pi4 - b, pi4 - a)
    else:
        return (pi4 - c, pi4 - b, a - pi4)


def accept_mirror(cost_u: int, cost_m: int, aggression: int) -> bool:
    """
    Algorithm 2 from the paper: decide whether to accept the mirror gate.

    aggression 0: never accept mirror
    aggression 1: accept only if strictly cheaper  (cost_m < cost_u)
    aggression 2: accept if cheaper or equal       (cost_m <= cost_u)  [default]
    aggression 3: always accept mirror
    """
    if aggression == 0:
        return False
    elif aggression == 1:
        return cost_m < cost_u
    elif aggression == 2:
        return cost_m <= cost_u
    elif aggression == 3:
        return True
    return False


def intermediate_layer_process(
    dag: DAGCircuit,
    aggression: int,
) -> DAGCircuit:
    """
    Walk the DAG and for each 2Q UnitaryGate node, check whether
    accepting its mirror reduces decomposition cost. If so, replace
    the gate with its mirror unitary in-place.

    This implements the intermediate layer from Section IV of the paper.
    Used standalone for testing; in the main pipeline this logic is
    incorporated into MirageSwap._mirror_pass().
    """
    for node in list(dag.topological_op_nodes()):
        if not isinstance(node.op, UnitaryGate):
            continue
        if len(node.qargs) != 2:
            continue

        U = Operator(node.op).data
        cost_u = decomp_cost(U)
        cost_m = decomp_cost(SWAP_MATRIX @ U)

        if accept_mirror(cost_u, cost_m, aggression):
            dag.substitute_node(
                node,
                UnitaryGate(SWAP_MATRIX @ U, check_input=False),
                inplace=True,
            )

    return dag
