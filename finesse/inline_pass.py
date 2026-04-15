# finesse/inline_pass.py
#
# Qiskit TransformationPass wrapping route() for pipeline integration.
#
# This is the canonical way to use route() in a Qiskit PassManager pipeline.
# route() is the single-trial core; InlineMirageSwap runs n_trials trials and
# selects the best by fidelity cost (when fidelity_matrix is provided) or gate
# depth otherwise.
#
# Ablation configurations reachable via mode, aggression, and fidelity_matrix:
#
#   SABRE:           mode='sabre',      aggression=0
#   LightSABRE:      mode='lightsabre', aggression=0
#   MIRAGE:          mode='lightsabre', aggression=2
#   MIRAGE+fid:      mode='lightsabre', aggression=2, fidelity_matrix=F
#   FINESSE:         mode='lightsabre', aggression=2, fidelity_matrix=F
#                    (FINESSE additionally uses FinesseLayout before this pass)
#
# Usage:
#   from finesse import InlineMirageSwap, make_unroll_consolidate, apply_trivial_layout
#
#   F_tuple = tuple(map(tuple, fidelity_matrix))   # must be tuple-of-tuples for MetaPass
#
#   pm = PassManager(
#       make_unroll_consolidate() + [
#           FidelityLayout(coupling_map, fidelity_matrix=F_tuple),
#           FullAncillaAllocation(coupling_map),
#           EnlargeWithAncilla(),
#           ApplyLayout(),
#           InlineMirageSwap(coupling_map, n_trials=20, seed=42,
#                            fidelity_matrix=F_tuple),
#           MirageDecompose(),
#           Optimize1qGatesDecomposition(basis=["rz", "ry", "rx"]),
#       ]
#   )

from __future__ import annotations

import copy
import numpy as np

from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler import CouplingMap
from qiskit.transpiler.basepasses import TransformationPass

from .routing import route
from .mirror import circuit_lf_cost


class InlineMirageSwap(TransformationPass):
    """
    Qiskit TransformationPass wrapping route() for pipeline integration.

    Runs n_trials independent routing trials (each with a different seed)
    and selects the best by:
      - Total log-infidelity cost (when fidelity_matrix is provided)
      - Gate depth otherwise

    All ablation configurations are reachable via mode, aggression,
    fidelity_matrix — see module docstring.

    Args:
        coupling_map:    Device connectivity.
        n_trials:        Number of independent routing trials (default 20).
        seed:            Base random seed; trial i uses seed + i.
        aggression:      Mirror acceptance level 0–3 (default 2):
                           0 = never mirror
                           1 = mirror only if strictly cheaper
                           2 = mirror if cheaper or equal  [default]
                           3 = always mirror
        mode:            Heuristic variant: 'lightsabre' (default) or 'sabre'.
        fidelity_matrix: Optional fidelity matrix as tuple-of-tuples (required
                         for MetaPass hashability). F[i][j] = fidelity of the
                         native 2Q gate on physical link (i,j). When provided,
                         H uses D_fid distances and the mirror layer compares
                         k_U·lf + H(cur) vs k_U'·lf + H(perm). Trials are
                         selected by minimum total log-infidelity cost.
        basis_gate:      Native 2Q gate for decomposition cost: 'sqrt_iswap'
                         (default) or 'cx'.
    """

    def __init__(
        self,
        coupling_map: CouplingMap,
        n_trials: int = 20,
        seed: int = 42,
        aggression: int = 2,
        mode: str = 'lightsabre',
        fidelity_matrix: tuple | None = None,
        basis_gate: str = 'sqrt_iswap',
    ):
        super().__init__()
        self.coupling_map    = coupling_map
        self.n_trials        = n_trials
        self.seed            = seed
        self.aggression      = aggression
        self.mode            = mode
        self.fidelity_matrix = fidelity_matrix   # tuple-of-tuples for hashability
        self.basis_gate      = basis_gate

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        F = np.array(self.fidelity_matrix) if self.fidelity_matrix is not None else None

        best_dag   = None
        best_score = float('inf')

        for trial_idx in range(self.n_trials):
            trial_dag, _, _ = route(
                copy.deepcopy(dag),
                self.coupling_map,
                aggression=self.aggression,
                seed=self.seed + trial_idx,
                mode=self.mode,
                fidelity_matrix=F,
                basis_gate=self.basis_gate,
            )

            score = (
                circuit_lf_cost(trial_dag, F, self.basis_gate)
                if F is not None
                else trial_dag.depth()
            )

            if score < best_score:
                best_score = score
                best_dag   = trial_dag

        return best_dag
