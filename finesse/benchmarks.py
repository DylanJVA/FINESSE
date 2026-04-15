# finesse/benchmarks.py
#
# Shared helpers for SABRE / LightSABRE / MIRAGE benchmarking and
# correctness checking.  Imported by notebooks and benchmark scripts.

from __future__ import annotations

import copy
import os
import warnings
from collections import Counter
from dataclasses import dataclass

import numpy as np

from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.quantum_info import Statevector, state_fidelity
from qiskit.transpiler import CouplingMap, PassManager
from qiskit.transpiler.passes import (
    TrivialLayout, FullAncillaAllocation, EnlargeWithAncilla, ApplyLayout,
    SabreSwap,
    HighLevelSynthesis, UnrollCustomDefinitions,
    BasisTranslator, Collect2qBlocks, ConsolidateBlocks,
)
from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary

from .routing import route  # re-exported as finesse.route
from .mirror import circuit_lf_cost


# ---------------------------------------------------------------------------
# Pipeline setup utilities
# ---------------------------------------------------------------------------

def make_unroll_consolidate():
    """
    Standard preprocessing pass sequence for MIRAGE / FINESSE.

    Decomposes high-level gates (qft, permutation, etc.) down to primitive
    u/cx/swap gates, then consolidates consecutive 2Q gates on the same qubit
    pair into single UnitaryGate nodes. The router sees only bare 2Q unitaries,
    which is what mirror absorption and decomposition expect.
    """
    return [
        HighLevelSynthesis(),
        UnrollCustomDefinitions(SessionEquivalenceLibrary,
                                basis_gates=["u", "cx", "swap"]),
        BasisTranslator(SessionEquivalenceLibrary,
                        target_basis=["u", "cx", "swap"]),
        Collect2qBlocks(),
        ConsolidateBlocks(),
    ]


def apply_trivial_layout(qc, coupling_map: CouplingMap):
    """Apply TrivialLayout to qc and return a physical DAGCircuit.

    Maps qubit i → physical qubit i, allocates ancilla qubits to fill
    the coupling map, and returns the physical DAG ready for routing.
    """
    pm = PassManager([
        TrivialLayout(coupling_map),
        FullAncillaAllocation(coupling_map),
        EnlargeWithAncilla(),
        ApplyLayout(),
    ])
    return circuit_to_dag(pm.run(qc))

# ---------------------------------------------------------------------------
# Q20 Tokyo coupling map (standard benchmark topology)
# ---------------------------------------------------------------------------
TOKYO_EDGES: list[list[int]] = [
    [0,1],[0,5],[1,2],[1,6],[1,7],[2,6],[3,8],[4,8],[4,9],
    [5,6],[5,10],[5,11],[6,7],[6,10],[6,11],[7,8],[7,12],
    [8,9],[8,12],[8,13],[10,11],[10,15],[11,12],[11,16],[11,17],
    [12,13],[12,16],[13,14],[13,18],[13,19],[14,18],[14,19],
    [15,16],[16,17],[17,18],
]

def make_tokyo() -> CouplingMap:
    """Return the Q20 Tokyo coupling map."""
    return CouplingMap(TOKYO_EDGES)


# ---------------------------------------------------------------------------
# Routing helper
# ---------------------------------------------------------------------------

def prepare_dag(qc_raw, coupling_map: CouplingMap):
    """
    Consolidate high-level gates and apply TrivialLayout.

    Returns (qc_consolidated, dag_physical) ready for route.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        qc_cons = PassManager(make_unroll_consolidate()).run(qc_raw.copy())
        dag = apply_trivial_layout(qc_cons.copy(), coupling_map)
    return qc_cons, dag



# ---------------------------------------------------------------------------
# Correctness helpers
# ---------------------------------------------------------------------------

def strip_regs(qc: QuantumCircuit) -> QuantumCircuit:
    """Return a copy of qc with all registers stripped (plain integer qubits)."""
    out = QuantumCircuit(qc.num_qubits)
    for instr in qc.data:
        qi = [qc.find_bit(q).index for q in instr.qubits]
        ci = [qc.find_bit(c).index for c in instr.clbits]
        out.append(instr.operation, qi, ci)
    return out


def permutation_correction_qc(n: int, cur: list[int]) -> QuantumCircuit:
    """
    Build a SWAP-network circuit that undoes the qubit permutation from routing.

    After routing, logical qubit ``log`` is at physical position ``cur[log]``.
    Composing the routed circuit with this correction and then comparing to the
    reference statevector (in logical-qubit order) gives fidelity 1 for a
    correct routing.

    Args:
        n:   Total number of qubits (physical).
        cur: cur[log] = physical qubit index after routing.

    Returns:
        A QuantumCircuit of ``n`` qubits whose SWAP gates undo the permutation.
    """
    perm_qc = QuantumCircuit(n)
    # phys_to_log[phys] = logical qubit currently at that physical position
    arr = [0] * n
    for log, phys in enumerate(cur):
        arr[phys] = log
    # Selection sort → minimal SWAP network to sort arr back to [0,1,...,n-1]
    for i in range(n):
        j = arr.index(i)
        if j != i:
            arr[i], arr[j] = arr[j], arr[i]
            perm_qc.swap(i, j)
    return perm_qc


def reference_statevector(qc_logical: QuantumCircuit, n_physical: int) -> Statevector:
    """
    Statevector of the logical circuit embedded in an n_physical-qubit space.

    Logical qubits 0..n_logical-1 map to physical 0..n_logical-1 (TrivialLayout).
    Physical qubits n_logical..n_physical-1 are ancillas, left in |0⟩.
    """
    ref_qc = QuantumCircuit(n_physical)
    for instr in qc_logical.data:
        qi = [qc_logical.find_bit(q).index for q in instr.qubits]
        ref_qc.append(instr.operation, qi)
    return Statevector(ref_qc)


def routing_fidelity(
    qc_logical: QuantumCircuit,
    coupling_map: CouplingMap,
    seed: int,
    mode: str = "lightsabre",
    aggression: int = 2,
    valve: bool | None = None,
) -> float:
    """
    Statevector fidelity of a single routed trial vs the reference.

    Returns a value in [0, 1]; 1.0 means the routing is correct.
    """
    n_phys = coupling_map.size()
    _, dag_in = prepare_dag(qc_logical, coupling_map)
    ref_sv = reference_statevector(qc_logical, n_phys)

    rd, _, cur = route(dag_in, coupling_map, seed=seed,
                       mode=mode, aggression=aggression, valve=valve)
    routed_qc = strip_regs(dag_to_circuit(rd))
    perm_qc = permutation_correction_qc(n_phys, cur)
    corrected = routed_qc.compose(perm_qc)
    return float(state_fidelity(Statevector(corrected), ref_sv))



# ---------------------------------------------------------------------------
# Benchmark (SWAP count) helpers
# ---------------------------------------------------------------------------

def swap_count(dag_routed) -> int:
    """Count SWAP gates in a routed DAG."""
    return dag_to_circuit(dag_routed).count_ops().get("swap", 0)


def benchmark_mode(
    dag_in,
    coupling_map: CouplingMap,
    seeds: list[int],
    mode: str,
    aggression: int,
    valve: bool | None = None,
) -> list[int]:
    """
    Run routing for each seed and return a list of SWAP counts.
    """
    counts = []
    for seed in seeds:
        rd, _, _ = route(dag_in, coupling_map, seed=seed,
                         mode=mode, aggression=aggression, valve=valve)
        counts.append(swap_count(rd))
    return counts


@dataclass(frozen=True)
class BenchmarkConfig:
    """
    A single transpilation configuration to benchmark.

    Ablation table rows (pass fidelity_matrix separately where needed):

      Row  name                  router          aggression  use_fidelity  fidelity_mirror
      1    SABRE                 sabre           0           False         —
      2    LightSABRE            lightsabre      0           False         —
      3    MIRAGE (SABRE)        mirage_sabre    2           False         —
      4    MIRAGE (LS)           mirage          2           False         —
      5    SABRE + fid           sabre           0           True          —
      6    LightSABRE + fid      lightsabre      0           True          —
      7    MIRAGE+fid (SABRE)    mirage_sabre    2           True          False
      8    MIRAGE+fid (LS)       mirage          2           True          False
      9    FINESSE               mirage          2           True          True

    Supported routers:
      - ``qiskit_sabre``: Qiskit's Rust SabreSwap (topology-only always)
      - ``sabre``:        our SABRE (aggression=0 by default)
      - ``lightsabre``:   our LightSABRE (aggression=0 by default)
      - ``mirage``:       LightSABRE + mirrors (aggression=2)
      - ``mirage_sabre``: SABRE + mirrors (aggression=2)

    fidelity_mirror controls whether fidelity enters the mirror acceptance
    (True = FINESSE unified metric; False = hop-count mirror, fidelity in H only).
    Ignored when use_fidelity=False or aggression=0.
    """

    name: str
    router: str
    use_fidelity: bool = False
    fidelity_mirror: bool = True
    trials: int = 1
    basis_gate: str = "sqrt_iswap"
    bidir_passes: int = 0
    aggression: int | None = None

    def route_kwargs(self, fidelity_matrix=None) -> dict:
        if self.router == "sabre":
            kwargs = {"mode": "sabre", "aggression": 0}
        elif self.router == "lightsabre":
            kwargs = {"mode": "lightsabre", "aggression": 0}
        elif self.router == "mirage":
            kwargs = {"mode": "lightsabre", "aggression": 2}
        elif self.router == "mirage_sabre":
            kwargs = {"mode": "sabre", "aggression": 2}
        else:
            raise ValueError(f"Unsupported router for route(): {self.router!r}")

        if self.aggression is not None:
            kwargs["aggression"] = self.aggression

        kwargs["basis_gate"] = self.basis_gate
        kwargs["bidir_passes"] = self.bidir_passes
        if self.use_fidelity and fidelity_matrix is not None:
            kwargs["fidelity_matrix"] = fidelity_matrix
            # Only pass fidelity_mirror when mirrors are active (aggression > 0)
            if kwargs.get("aggression", 0) > 0:
                kwargs["fidelity_mirror"] = self.fidelity_mirror
        return kwargs


def _normalize_circuit_entry(entry) -> tuple[str, str]:
    if isinstance(entry, tuple):
        if len(entry) != 2:
            raise ValueError(f"Circuit entries must be (label, qasm_name), got {entry!r}")
        return entry
    if isinstance(entry, str):
        label = entry[:-5] if entry.endswith(".qasm") else entry
        qasm_name = entry if entry.endswith(".qasm") else f"{entry}.qasm"
        return label, qasm_name
    raise TypeError(f"Unsupported circuit entry: {entry!r}")


def _qiskit_sabre_trial(qc, cm, *, seed: int):
    pm = PassManager([
        TrivialLayout(cm),
        FullAncillaAllocation(cm),
        EnlargeWithAncilla(),
        ApplyLayout(),
        SabreSwap(cm, heuristic="decay", seed=seed, trials=1),
    ])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pm.run(qc.copy())


def _run_trial_config(
    qc,
    dag_phys,
    coupling_map: CouplingMap,
    config: BenchmarkConfig,
    *,
    seed: int,
    fidelity_matrix=None,
):
    use_fidelity = config.use_fidelity and fidelity_matrix is not None

    if config.router == "qiskit_sabre":
        routed_qc = _qiskit_sabre_trial(qc, coupling_map, seed=seed)
        routed_dag = circuit_to_dag(routed_qc)
        score = (
            circuit_lf_cost(routed_dag, fidelity_matrix, config.basis_gate)
            if use_fidelity
            else routed_qc.depth()
        )
        return routed_dag, score

    routed_dag, _, _ = route(
        copy.deepcopy(dag_phys),
        coupling_map,
        seed=seed,
        **config.route_kwargs(fidelity_matrix if use_fidelity else None),
    )
    score = (
        circuit_lf_cost(routed_dag, fidelity_matrix, config.basis_gate)
        if use_fidelity
        else dag_to_circuit(routed_dag).depth()
    )
    return routed_dag, score


def _run_trial_config_with_mapping(
    qc,
    dag_phys,
    coupling_map: CouplingMap,
    config: BenchmarkConfig,
    *,
    seed: int,
    fidelity_matrix=None,
):
    use_fidelity = config.use_fidelity and fidelity_matrix is not None

    if config.router == "qiskit_sabre":
        routed_qc = _qiskit_sabre_trial(qc, coupling_map, seed=seed)
        routed_dag = circuit_to_dag(routed_qc)
        layout = getattr(routed_qc, "layout", None)
        final_cur = list(layout.final_index_layout()) if layout is not None else list(range(routed_qc.num_qubits))
        score = (
            circuit_lf_cost(routed_dag, fidelity_matrix, config.basis_gate)
            if use_fidelity
            else routed_qc.depth()
        )
        return routed_dag, final_cur, score

    routed_dag, _, final_cur = route(
        copy.deepcopy(dag_phys),
        coupling_map,
        seed=seed,
        **config.route_kwargs(fidelity_matrix if use_fidelity else None),
    )
    score = (
        circuit_lf_cost(routed_dag, fidelity_matrix, config.basis_gate)
        if use_fidelity
        else dag_to_circuit(routed_dag).depth()
    )
    return routed_dag, final_cur, score


def run_benchmark(
    configs: list[BenchmarkConfig],
    circuits: list[tuple[str, str] | str],
    *,
    coupling_map: CouplingMap,
    n_seeds: int = 10,
    fidelity_matrix=None,
    verbose: bool = True,
):
    """
    Benchmark arbitrary transpilation configurations on a chosen circuit set.

    Args:
        configs:          List of BenchmarkConfig entries.
        circuits:         Circuit names or ``(label, qasm_name)`` pairs.
        coupling_map:     Target topology.
        n_seeds:          Independent seeds per configuration.
        fidelity_matrix:  Optional fidelity matrix used when
                          ``config.use_fidelity`` is True.
        verbose:          Print per-circuit progress.

    Returns:
        A pandas DataFrame when pandas is installed, otherwise a list of row
        dicts with one entry per (circuit, config, seed).
    """
    rows: list[dict] = []

    for circuit_entry in circuits:
        circ_label, qasm_name = _normalize_circuit_entry(circuit_entry)
        if verbose:
            print(f"[{circ_label}]")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            qc = fetch_qasm(qasm_name)
            dag_phys = apply_trivial_layout(qc.copy(), coupling_map)

        for config in configs:
            swap_list = []
            depth_list = []
            lf_list = []

            for seed in range(n_seeds):
                best_dag = None
                best_score = float("inf")

                for trial in range(config.trials):
                    trial_seed = seed * max(config.trials, 1) + trial
                    trial_dag, score = _run_trial_config(
                        qc,
                        dag_phys,
                        coupling_map,
                        config,
                        seed=trial_seed,
                        fidelity_matrix=fidelity_matrix,
                    )
                    if score < best_score:
                        best_score = score
                        best_dag = trial_dag

                routed_qc = dag_to_circuit(best_dag)
                swap_count_value = routed_qc.count_ops().get("swap", 0)
                depth_value = routed_qc.depth()
                lf_value = (
                    circuit_lf_cost(best_dag, fidelity_matrix, config.basis_gate)
                    if fidelity_matrix is not None
                    else float("nan")
                )

                swap_list.append(swap_count_value)
                depth_list.append(depth_value)
                lf_list.append(lf_value)

                rows.append({
                    "circuit": circ_label,
                    "qasm_name": qasm_name,
                    "config": config.name,
                    "router": config.router,
                    "use_fidelity": config.use_fidelity,
                    "trials": config.trials,
                    "aggression": config.aggression,
                    "seed": seed,
                    "swap_count": swap_count_value,
                    "gate_depth": depth_value,
                    "lf_cost": lf_value,
                    "fidelity": float(np.exp(-lf_value)) if np.isfinite(lf_value) else float("nan"),
                })

            if verbose:
                avg_sw = float(np.mean(swap_list))
                avg_depth = float(np.mean(depth_list))
                if fidelity_matrix is not None:
                    avg_lf = float(np.nanmean(lf_list))
                    print(
                        f"  {config.name:<24} swaps={avg_sw:.1f}  depth={avg_depth:.1f}  -logF={avg_lf:.2f}"
                    )
                else:
                    print(f"  {config.name:<24} swaps={avg_sw:.1f}  depth={avg_depth:.1f}")

    try:
        import pandas as pd
    except ModuleNotFoundError:
        return rows
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Verbose correctness check
# ---------------------------------------------------------------------------

def check_unitary_equivalence(
    qc1: QuantumCircuit,
    qc2: QuantumCircuit,
    *,
    max_qubits: int = 8,
) -> bool | None:
    """
    Check if two circuits implement the same unitary up to global phase.

    Uses Operator (full 2^n × 2^n unitary matrix). Exponentially expensive —
    only practical for small circuits. Returns None if num_qubits > max_qubits
    rather than raising, so callers can handle it gracefully.

    For routing correctness checks, compose permutation_correction_qc onto the
    routed circuit before calling, or use check_routing which handles this
    automatically via statevector comparison.

    Args:
        qc1, qc2:   Circuits to compare. Must have the same number of qubits.
        max_qubits: Skip and return None above this size. Default 8.

    Returns:
        True if unitarily equivalent, False if not, None if too large to check.
    """
    from qiskit.quantum_info import Operator as _Op
    if qc1.num_qubits != qc2.num_qubits:
        raise ValueError(
            f"Qubit count mismatch: {qc1.num_qubits} vs {qc2.num_qubits}"
        )
    if qc1.num_qubits > max_qubits:
        return None
    return _Op(qc1).equiv(_Op(qc2))


# Gates that form a valid Clifford circuit (closed under composition with SWAP)
_CLIFFORD_GATE_NAMES: frozenset[str] = frozenset(
    {'h', 's', 'sdg', 'x', 'y', 'z', 'cx', 'cz', 'swap', 'id', 'sx', 'sxdg', 'cy'}
)


def _is_clifford_circuit(qc: QuantumCircuit) -> bool:
    """Return True if every non-barrier/measure/reset gate in qc is a Clifford gate."""
    for inst in qc.data:
        name = inst.operation.name
        if name in ('barrier', 'measure', 'reset'):
            continue
        if name not in _CLIFFORD_GATE_NAMES:
            return False
    return True


def random_clifford_circuit(
    n_qubits: int,
    n_gates: int,
    seed: int = 0,
) -> QuantumCircuit:
    """
    Build a random Clifford circuit suitable for large-scale routing correctness tests.

    Uses only h, s, x, y, z (1Q) and cx (2Q) gates.  Approximately 40% of
    instructions are cx gates.  All gates are Clifford, so the circuit can be
    verified with qiskit.quantum_info.Clifford at any qubit count.

    Args:
        n_qubits: Number of logical qubits.
        n_gates:  Total instruction count (1Q + 2Q).
        seed:     RNG seed for reproducibility.
    """
    rng = np.random.default_rng(seed)
    qc  = QuantumCircuit(n_qubits)
    one_q = ['h', 's', 'x', 'y', 'z']
    for _ in range(n_gates):
        if n_qubits >= 2 and rng.random() < 0.4:
            i, j = rng.choice(n_qubits, size=2, replace=False)
            qc.cx(int(i), int(j))
        else:
            gate = one_q[int(rng.integers(len(one_q)))]
            q    = int(rng.integers(n_qubits))
            getattr(qc, gate)(q)
    return qc


def _gate_summary(qc: QuantumCircuit):
    n2q = 0; n_meas = 0; by_1q: Counter = Counter()
    for inst in qc.data:
        op = inst.operation
        if op.name in ('barrier', 'swap'): continue
        if op.name == 'measure':          n_meas += 1
        elif op.num_qubits >= 2:          n2q += 1
        else:                             by_1q[op.name] += 1
    return n2q, n_meas, by_1q


def _strip_nonunitary_ops(qc: QuantumCircuit) -> QuantumCircuit:
    out = QuantumCircuit(qc.num_qubits)
    for inst in qc.data:
        if inst.operation.name not in ('measure', 'reset', 'barrier'):
            qi = [qc.find_bit(q).index for q in inst.qubits]
            out.append(inst.operation, qi, [])
    return out


def evaluate_routing_checks(
    original_qc: QuantumCircuit,
    routed_qc: QuantumCircuit,
    final_cur: list[int],
    coupling_map_edges: list,
    *,
    dag_phys=None,
    run_statevector: bool = True,
    run_unitary: bool = True,
    run_clifford: bool = True,
    sv_max_qubits: int = 15,
    unitary_max_qubits: int = 8,
) -> dict[str, object]:
    """
    Run the full set of routing correctness checks and return structured results.

    Checks:
      - adjacency
      - gate counts
      - statevector equivalence (when enabled and small enough)
      - unitary equivalence (when enabled and small enough)
      - Clifford equivalence (when enabled and the original circuit is Clifford)

    Status values are:
      - True: passed
      - False: failed
      - None: skipped / not applicable
    """
    from qiskit.converters import dag_to_circuit as _d2c

    results: dict[str, object] = {
        "adjacency": None,
        "gate_counts": None,
        "statevector": None,
        "unitary": None,
        "clifford": None,
        "messages": [],
    }

    messages: list[str] = []
    cm_set = {(min(a, b), max(a, b)) for a, b in coupling_map_edges}

    # Adjacency
    violations = []
    for inst in routed_qc.data:
        if inst.operation.num_qubits != 2:
            continue
        q0 = routed_qc.find_bit(inst.qubits[0]).index
        q1 = routed_qc.find_bit(inst.qubits[1]).index
        if (min(q0, q1), max(q0, q1)) not in cm_set:
            violations.append((q0, q1, inst.operation.name))
    if violations:
        results["adjacency"] = False
        messages.append(f"FAIL adjacency: {violations[:3]}{'...' if len(violations) > 3 else ''}")
    else:
        results["adjacency"] = True
        messages.append("OK   adjacency")

    # Gate counts
    ref_qc = _d2c(dag_phys) if dag_phys is not None else original_qc
    routing_swaps = routed_qc.count_ops().get('swap', 0)
    ref_2q, ref_meas, ref_1q = _gate_summary(ref_qc)
    rout_2q, rout_meas, rout_1q = _gate_summary(routed_qc)
    gate_ok = True
    if ref_2q != rout_2q:
        messages.append(f"FAIL gate counts: 2Q ref={ref_2q} routed={rout_2q}  (+{routing_swaps} SWAPs)")
        gate_ok = False
    if ref_meas != rout_meas:
        messages.append(f"FAIL gate counts: measurements ref={ref_meas} routed={rout_meas}")
        gate_ok = False
    if ref_1q != rout_1q:
        missing = {k: ref_1q[k] - rout_1q.get(k, 0) for k in ref_1q if ref_1q[k] != rout_1q.get(k, 0)}
        extra   = {k: rout_1q[k] - ref_1q.get(k, 0) for k in rout_1q if rout_1q[k] != ref_1q.get(k, 0)}
        messages.append(f"FAIL gate counts: 1Q missing={missing} extra={extra}")
        gate_ok = False
    if gate_ok:
        messages.append(f"OK   gate counts  (+{routing_swaps} routing SWAPs)")
    results["gate_counts"] = gate_ok

    n = routed_qc.num_qubits
    ref_bare = _strip_nonunitary_ops(ref_qc)
    rout_bare = _strip_nonunitary_ops(routed_qc)
    perm_qc = permutation_correction_qc(n, final_cur)
    corrected = rout_bare.compose(perm_qc)

    # Clifford equivalence for all Clifford circuits, regardless of size
    if run_clifford and _is_clifford_circuit(original_qc):
        from qiskit.quantum_info import Clifford as _Clifford
        try:
            cliff_ref = _Clifford(ref_bare)
            cliff_rout = _Clifford(corrected)
            if cliff_ref != cliff_rout:
                results["clifford"] = False
                messages.append("FAIL clifford mismatch")
            else:
                results["clifford"] = True
                messages.append("OK   clifford")
        except Exception as e:
            results["clifford"] = None
            messages.append(f"SKIP clifford (error: {e})")

    # Statevector equivalence for small enough circuits
    if run_statevector:
        if n > sv_max_qubits:
            messages.append(f"SKIP statevector ({n} qubits > {sv_max_qubits})")
        else:
            from qiskit.quantum_info import Statevector as _SV
            if not _SV(corrected).equiv(_SV(ref_bare)):
                results["statevector"] = False
                messages.append("FAIL statevector mismatch")
            else:
                results["statevector"] = True
                messages.append("OK   statevector")

    # Unitary equivalence for small enough circuits
    if run_unitary:
        result = check_unitary_equivalence(ref_bare, corrected, max_qubits=unitary_max_qubits)
        if result is None:
            messages.append(f"SKIP unitary ({n} qubits > {unitary_max_qubits})")
        elif result:
            results["unitary"] = True
            messages.append("OK   unitary")
        else:
            results["unitary"] = False
            messages.append("FAIL unitary mismatch")

    results["messages"] = messages
    results["ok"] = all(
        value is not False
        for key, value in results.items()
        if key not in {"messages", "ok"}
    )
    return results


def print_routing_check_report(results: dict[str, object], *, label: str = "") -> None:
    tag = f"[{label}] " if label else ""
    for message in results["messages"]:
        print(f"{tag}{message}")


def check_routing(
    original_qc: QuantumCircuit,
    routed_qc: QuantumCircuit,
    final_cur: list[int],
    coupling_map_edges: list,
    *,
    dag_phys=None,
    label: str = "",
    verify: str | None = 'statevector',
    sv_max_qubits: int = 15,
) -> bool:
    """
    Verbose correctness check for a single routed circuit.

    Runs adjacency and gate-count checks, then prints whichever equivalence
    checks are requested.

    Args:
        original_qc:        Unrouted logical circuit.
        routed_qc:          Circuit returned by routing (dag_to_circuit(rd)).
        final_cur:          Final qubit permutation from route.
        coupling_map_edges: List of [a, b] pairs defining allowed 2Q connections.
        dag_phys:           Physical DAG after layout (used as gate-count reference
                            when ancilla qubits are present).
        label:              Prefix for printed lines, e.g. 'sym9/sabre/trial0'.
        verify:             Equivalence check mode:
                              'statevector' (default) — Statevector on |0⟩, skipped
                                  if > 15 qubits. Fast, sufficient for routing checks.
                              'unitary' — Full Operator comparison via
                                  check_unitary_equivalence, skipped if > 8 qubits.
                                  Stronger but exponentially more expensive.
                              'all' — run statevector, unitary, and Clifford checks
                                  when applicable.
                              None — skip equivalence checks entirely.

    Returns:
        True if all checks pass.

    Example output::

        [sym9/sabre/trial0] OK   adjacency
        [sym9/sabre/trial0] OK   gate counts  (+12 routing SWAPs)
        [sym9/sabre/trial0] OK   statevector
    """
    if verify not in ('statevector', 'unitary', 'all', None):
        raise ValueError(f"verify must be 'statevector', 'unitary', 'all', or None; got {verify!r}")

    results = evaluate_routing_checks(
        original_qc,
        routed_qc,
        final_cur,
        coupling_map_edges,
        dag_phys=dag_phys,
        run_statevector=(verify in {'statevector', 'all'}),
        run_unitary=(verify in {'unitary', 'all'}),
        run_clifford=(verify == 'all'),
        sv_max_qubits=sv_max_qubits,
    )
    if verify is None:
        results["messages"].append("SKIP equivalence (disabled)")
    print_routing_check_report(results, label=label)
    return bool(results["ok"])


# ---------------------------------------------------------------------------
# Standard correctness suite
# ---------------------------------------------------------------------------

_CIRCUITS_DIR    = os.path.join(os.path.dirname(os.path.dirname(__file__)), "circuits")
_REDQUEEN_DIR    = os.path.join(_CIRCUITS_DIR, "redqueen")
_QASMBENCH_DIR   = os.path.join(_CIRCUITS_DIR, "qasmbench")
_DATA_DIR        = os.path.join(os.path.dirname(__file__), "data")


def _load_qasm(path: str) -> QuantumCircuit:
    from qiskit.qasm2 import loads as _qasm2_loads, CustomInstruction
    from qiskit.circuit.library import SXGate, SXdgGate
    custom = [
        CustomInstruction("sx",   0, 1, SXGate,   builtin=True),
        CustomInstruction("sxdg", 0, 1, SXdgGate, builtin=True),
    ]
    with open(path) as f:
        return _qasm2_loads(f.read(), custom_instructions=custom)


def _with_qasm_suffix(name: str) -> str:
    return name if name.endswith(".qasm") else f"{name}.qasm"


def available_redqueen_circuits() -> list[str]:
    """Return locally available red-queen circuit filenames."""
    names: set[str] = set()
    for directory in (_DATA_DIR, _REDQUEEN_DIR):
        if not os.path.isdir(directory):
            continue
        names.update(
            name for name in os.listdir(directory)
            if name.endswith(".qasm")
        )
    return sorted(names)


def available_qasmbench_circuits(size: str = "small") -> list[str]:
    """Return locally available QASMBench circuit names for the given size bucket."""
    root = os.path.join(_QASMBENCH_DIR, size)
    if not os.path.isdir(root):
        return []
    out = []
    for name in os.listdir(root):
        qasm_path = os.path.join(root, name, f"{name}.qasm")
        if os.path.exists(qasm_path):
            out.append(name)
    return sorted(out)


def fetch_qasm(name: str) -> QuantumCircuit:
    """Load a red-queen circuit from the local cache.

    Checks the bundled data/ directory first (correctness suite circuits are
    always available), then the downloaded circuits/redqueen/ cache.
    Run ``finesse-download`` to populate the full suite.
    """
    name = _with_qasm_suffix(name)
    for directory in (_DATA_DIR, _REDQUEEN_DIR):
        path = os.path.join(directory, name)
        if os.path.exists(path):
            return _load_qasm(path)
    examples = ", ".join(available_redqueen_circuits()[:5])
    raise FileNotFoundError(
        f"Red-queen circuit not found: {name}\n"
        "Run `finesse-download --source redqueen` to populate the local cache.\n"
        "If you meant a QASMBench circuit such as `adder_n10`, use "
        "`fetch_qasmbench(...)` instead.\n"
        f"Example local red-queen names: {examples if examples else '[none found locally]'}"
    )


def fetch_qasmbench(name: str, size: str = "small") -> QuantumCircuit:
    """Load a QASMBench circuit from the local cache.

    Run ``finesse-download --source qasmbench`` to populate.
    ``name`` is the circuit directory name, e.g. ``'adder_n10'``.
    ``size`` is ``'small'``, ``'medium'``, or ``'large'``.
    """
    name = name.removesuffix(".qasm")
    path = os.path.join(_QASMBENCH_DIR, size, name, f"{name}.qasm")
    if not os.path.exists(path):
        examples = ", ".join(available_qasmbench_circuits(size)[:5])
        raise FileNotFoundError(
            f"Circuit not found: {path}\n"
            "Run `finesse-download --source qasmbench` to download QASMBench circuits.\n"
            f"Example local qasmbench/{size} names: {examples if examples else '[none found locally]'}"
        )
    return _load_qasm(path)


def load_redqueen_circuits(names: list[str] | None = None) -> dict[str, QuantumCircuit]:
    """
    Load a batch of red-queen circuits from the local cache.

    If ``names`` is omitted, loads every locally available red-queen circuit.
    This is useful when you want circuit acquisition to happen once up front
    before benchmarking, instead of calling ``fetch_qasm`` repeatedly.
    """
    if names is None:
        names = available_redqueen_circuits()
    return {name.removesuffix(".qasm"): fetch_qasm(name) for name in names}


def load_qasmbench_circuits(
    names: list[str] | None = None,
    *,
    size: str = "small",
) -> dict[str, QuantumCircuit]:
    """
    Load a batch of QASMBench circuits from the local cache.

    If ``names`` is omitted, loads every locally available circuit in the
    requested size bucket.
    """
    if names is None:
        names = available_qasmbench_circuits(size)
    return {name.removesuffix(".qasm"): fetch_qasmbench(name, size=size) for name in names}


def line_cm(n: int) -> CouplingMap:
    """Linear chain coupling map: 0–1–2–…–(n-1)."""
    return CouplingMap([[i, i + 1] for i in range(n - 1)])


def grid_cm(rows: int, cols: int) -> CouplingMap:
    """2D rectangular grid coupling map."""
    edges = []
    for i in range(rows):
        for j in range(cols):
            q = i * cols + j
            if j + 1 < cols: edges.append([q, q + 1])
            if i + 1 < rows: edges.append([q, q + cols])
    return CouplingMap(edges)


# private aliases used internally
_line_cm = line_cm
_grid_cm = grid_cm


_CORRECTNESS_SUITE = [
    ('rd32-v0_66/line4',   'rd32-v0_66.qasm',  _line_cm(4),
     [[i, i+1] for i in range(3)]),
    ('4gt11_84/line5',     '4gt11_84.qasm',    _line_cm(5),
     [[i, i+1] for i in range(4)]),
    ('ham7_104/grid3x3',   'ham7_104.qasm',    _grid_cm(3, 3),
     [[i*3+j, i*3+j+1] for i in range(3) for j in range(2)] +
     [[i*3+j, (i+1)*3+j] for i in range(2) for j in range(3)]),
    ('rd53_135/grid3x3',   'rd53_135.qasm',    _grid_cm(3, 3),
     [[i*3+j, i*3+j+1] for i in range(3) for j in range(2)] +
     [[i*3+j, (i+1)*3+j] for i in range(2) for j in range(3)]),
    ('rd84_142/line15',    'rd84_142.qasm',    _line_cm(15),
     [[i, i+1] for i in range(14)]),
]


CORRECTNESS_CIRCUIT_LABELS: list[str] = [label for label, *_ in _CORRECTNESS_SUITE]
"""Labels for the circuits in the standard correctness suite.

Available labels:
  'rd32-v0_66/line4'   — 4-qubit line topology
  '4gt11_84/line5'     — 5-qubit line topology
  'ham7_104/grid3x3'   — 3×3 grid topology
  'rd53_135/grid3x3'   — 3×3 grid topology
  'rd84_142/line15'    — 15-qubit line topology
"""


def run_correctness_suite(
    modes: list[str] | None = None,
    n_trials: int = 5,
    aggression: int = 0,
    circuits: list[str] | None = None,
    verify: str | None = 'statevector',
    sv_max_qubits: int = 15,
) -> bool:
    """
    Run the standard correctness suite across routing modes.

    Routes circuits on compact topologies, checks adjacency, gate counts,
    and optionally statevector equivalence for each trial.

    Args:
        modes:      List of mode strings to test. Defaults to
                    ['sabre', 'lightsabre'].
        n_trials:   Number of trials (seeds 0..n_trials-1) per circuit.
        aggression: Mirror absorption level passed to route.
                    Use 0 to test routing only, >0 to include MIRAGE.
        circuits:   Subset of circuit labels to run. Defaults to all five.
                    See CORRECTNESS_CIRCUIT_LABELS for available names.
        verify:     Equivalence check mode passed to check_routing:
                    'statevector' (default), 'unitary', or None to skip.

    Returns:
        True if every check passes.

    Example::

        from finesse import run_correctness_suite
        run_correctness_suite()
        # [rd32-v0_66/line4/sabre/trial0] OK   adjacency
        # [rd32-v0_66/line4/sabre/trial0] OK   gate counts  (+2 routing SWAPs)
        # [rd32-v0_66/line4/sabre/trial0] OK   statevector
        # ...
        # All checks passed ✓
    """
    if modes is None:
        modes = ['sabre', 'lightsabre']

    suite = _CORRECTNESS_SUITE
    if circuits is not None:
        unknown = set(circuits) - set(CORRECTNESS_CIRCUIT_LABELS)
        if unknown:
            raise ValueError(f"Unknown circuit labels: {unknown}. "
                             f"Available: {CORRECTNESS_CIRCUIT_LABELS}")
        suite = [entry for entry in _CORRECTNESS_SUITE if entry[0] in circuits]

    all_ok = True
    for mode in modes:
        print(f"--- mode={mode} ---")
        for circ_label, qasm_name, cm, cm_edges in suite:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                qc = fetch_qasm(qasm_name)
                dag_phys = apply_trivial_layout(qc.copy(), cm)
            circuit_ok = True
            for trial in range(n_trials):
                rd, _, fc = route(
                    copy.deepcopy(dag_phys), cm,
                    seed=trial, mode=mode, aggression=aggression,
                )
                rqc = dag_to_circuit(rd)
                ok = check_routing(qc, rqc, fc, cm_edges,
                                   dag_phys=dag_phys,
                                   label=f'{circ_label}/{mode}/trial{trial}',
                                   verify=verify,
                                   sv_max_qubits=sv_max_qubits)
                circuit_ok = circuit_ok and ok
                all_ok = all_ok and ok
            status = 'OK' if circuit_ok else 'FAIL'
            print(f"  {circ_label}: {status} ({n_trials} trials)")
        print()

    print('All checks passed \u2713' if all_ok else 'SOME CHECKS FAILED \u2717')
    return all_ok


def run_config_correctness_suite(
    config: BenchmarkConfig,
    *,
    standard_circuits: list[str] | None = None,
    clifford_qubit_counts: list[int] | None = None,
    clifford_gate_multipliers: list[int] | None = None,
    n_seeds: int = 3,
    fidelity_matrix=None,
    sv_max_qubits: int = 15,
    unitary_max_qubits: int = 8,
    verbose: bool = True,
):
    """
    Run a broad correctness sweep for a single BenchmarkConfig.

    This is the config-oriented counterpart to the older paper helpers. For
    each routed circuit it runs:
      - adjacency
      - gate count preservation
      - statevector equivalence (small enough circuits)
      - unitary equivalence (small enough circuits)
      - Clifford equivalence (for any Clifford circuit, regardless of size)

    The suite combines:
      - the bundled small-topology correctness circuits
      - randomly generated Clifford circuits on grid topologies

    Returns a pandas DataFrame when pandas is available, otherwise a list of
    result rows.
    """
    if clifford_qubit_counts is None:
        clifford_qubit_counts = [10, 20, 40]
    if clifford_gate_multipliers is None:
        clifford_gate_multipliers = [3, 10]

    rows: list[dict] = []

    # Standard bundled circuits on their native small coupling maps
    suite = _CORRECTNESS_SUITE
    if standard_circuits is not None:
        unknown = set(standard_circuits) - set(CORRECTNESS_CIRCUIT_LABELS)
        if unknown:
            raise ValueError(f"Unknown circuit labels: {unknown}. Available: {CORRECTNESS_CIRCUIT_LABELS}")
        suite = [entry for entry in _CORRECTNESS_SUITE if entry[0] in standard_circuits]

    for circ_label, qasm_name, cm_small, cm_edges in suite:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            qc = fetch_qasm(qasm_name)
            dag_phys = apply_trivial_layout(qc.copy(), cm_small)

        for seed in range(n_seeds):
            best_dag = None
            best_cur = None
            best_score = float("inf")
            for trial in range(config.trials):
                trial_seed = seed * max(config.trials, 1) + trial
                trial_dag, trial_cur, score = _run_trial_config_with_mapping(
                    qc,
                    dag_phys,
                    cm_small,
                    config,
                    seed=trial_seed,
                    fidelity_matrix=fidelity_matrix,
                )
                if score < best_score:
                    best_dag = trial_dag
                    best_cur = trial_cur
                    best_score = score

            routed_qc = dag_to_circuit(best_dag)
            label = f"{config.name}/{circ_label}/s{seed}"
            results = evaluate_routing_checks(
                qc,
                routed_qc,
                best_cur,
                cm_edges,
                dag_phys=dag_phys,
                run_statevector=True,
                run_unitary=True,
                run_clifford=True,
                sv_max_qubits=sv_max_qubits,
                unitary_max_qubits=unitary_max_qubits,
            )
            if verbose:
                print_routing_check_report(results, label=label)

            row = {
                "suite": "standard",
                "label": circ_label,
                "config": config.name,
                "seed": seed,
                "adjacency": results["adjacency"],
                "gate_counts": results["gate_counts"],
                "statevector": results["statevector"],
                "unitary": results["unitary"],
                "clifford": results["clifford"],
                "ok": results["ok"],
            }
            rows.append(row)

    # Random Clifford circuits on grids
    def _best_grid(n: int) -> CouplingMap:
        cols = int(np.ceil(np.sqrt(n)))
        rows_ = int(np.ceil(n / cols))
        return grid_cm(rows_, cols)

    for n_qubits in clifford_qubit_counts:
        cm = _best_grid(n_qubits)
        cm_edges = list(cm.get_edges())
        F_cfg = fidelity_matrix
        if config.use_fidelity and F_cfg is None:
            from .ablation import make_synthetic_fidelity
            F_cfg = make_synthetic_fidelity(cm, seed=42)

        for mult in clifford_gate_multipliers:
            n_gates = n_qubits * mult
            circ_label = f"clifford_{n_qubits}q_{n_gates}g"
            for seed in range(n_seeds):
                qc = random_clifford_circuit(n_qubits, n_gates, seed=seed)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    dag_phys = apply_trivial_layout(qc.copy(), cm)

                best_dag = None
                best_cur = None
                best_score = float("inf")
                for trial in range(config.trials):
                    trial_seed = seed * max(config.trials, 1) + trial
                    trial_dag, trial_cur, score = _run_trial_config_with_mapping(
                        qc,
                        dag_phys,
                        cm,
                        config,
                        seed=trial_seed,
                        fidelity_matrix=F_cfg,
                    )
                    if score < best_score:
                        best_dag = trial_dag
                        best_cur = trial_cur
                        best_score = score

                routed_qc = dag_to_circuit(best_dag)
                label = f"{config.name}/{circ_label}/s{seed}"
                results = evaluate_routing_checks(
                    qc,
                    routed_qc,
                    best_cur,
                    cm_edges,
                    dag_phys=dag_phys,
                    run_statevector=True,
                    run_unitary=True,
                    run_clifford=True,
                    sv_max_qubits=sv_max_qubits,
                    unitary_max_qubits=unitary_max_qubits,
                )
                if verbose:
                    print_routing_check_report(results, label=label)

                rows.append({
                    "suite": "clifford",
                    "label": circ_label,
                    "config": config.name,
                    "seed": seed,
                    "adjacency": results["adjacency"],
                    "gate_counts": results["gate_counts"],
                    "statevector": results["statevector"],
                    "unitary": results["unitary"],
                    "clifford": results["clifford"],
                    "ok": results["ok"],
                })

    if verbose:
        n_ok = sum(1 for row in rows if row["ok"])
        print(f"\n{config.name}: {n_ok}/{len(rows)} routed cases passed all applicable checks")

    try:
        import pandas as pd
    except ModuleNotFoundError:
        return rows
    return pd.DataFrame(rows)


def run_clifford_correctness_suite(
    qubit_counts: list[int] | None = None,
    gate_multipliers: list[int] | None = None,
    n_seeds: int = 10,
    modes: list[str] | None = None,
    configs: dict[str, dict] | None = None,
    sv_max_qubits: int = 15,
) -> bool:
    """
    Scalability correctness check using random Clifford circuits.

    Generates Clifford circuits at several qubit counts and gate densities,
    routes each through the specified modes/configs, and verifies correctness
    with Clifford comparison (which scales to hundreds of qubits in seconds).

    Circuits with n ≤ sv_max_qubits fall back to the statevector check;
    larger circuits use Clifford comparison.  All circuits are Clifford by
    construction so the Clifford branch is always exercised for large sizes.

    Args:
        qubit_counts:      Physical qubit counts to test. Grid topologies are
                           chosen to cover each count (e.g. 30 → grid(5,6)).
        gate_multipliers:  Gate counts as multiples of n_qubits. Default [3, 10].
        n_seeds:           Seeds per (size, density, config) combination.
        modes:             Shorthand routing modes: 'sabre', 'lightsabre',
                           'mirage' (lightsabre + aggression=2). Used when
                           ``configs`` is not provided. Default: all three.
        configs:           Explicit routing configs as {label: kwargs_for_route}.
                           When provided, ``modes`` is ignored.  A fidelity_matrix
                           key in kwargs is replaced per-circuit with a synthetic
                           matrix of the appropriate size (use sentinel value True
                           for synthetic F, False/None for uniform F, or supply a
                           fixed array).  Mutually exclusive with ``modes``.
        sv_max_qubits:     Threshold for statevector vs Clifford check (default 15).

    Returns:
        True if all checks pass.

    Example — test all ablation configs including finesse::

        from finesse.benchmarks import run_clifford_correctness_suite
        from finesse.ablation import ABLATION_CONFIGS, FIDELITY_CONFIGS

        configs = {}
        for key, kwargs in ABLATION_CONFIGS.items():
            kw = dict(kwargs)
            if key in FIDELITY_CONFIGS:
                kw['fidelity_matrix'] = True   # replaced with synthetic F per circuit
            configs[key] = kw
        run_clifford_correctness_suite(configs=configs, n_seeds=5)
    """
    if qubit_counts is None:
        qubit_counts = [30, 50, 80]
    if gate_multipliers is None:
        gate_multipliers = [3, 10]

    # Build the config table
    if configs is not None:
        cfg_table = configs
    else:
        if modes is None:
            modes = ['sabre', 'lightsabre', 'mirage']
        cfg_table = {}
        for mode in modes:
            if mode == 'mirage':
                cfg_table[mode] = {'mode': 'lightsabre', 'aggression': 2}
            else:
                cfg_table[mode] = {'mode': mode, 'aggression': 0}

    def _best_grid(n: int) -> CouplingMap:
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))
        return grid_cm(rows, cols)

    all_ok = True
    for n_qubits in qubit_counts:
        cm = _best_grid(n_qubits)
        n_phys = cm.size()
        cm_edges = list(cm.get_edges())
        # Build fidelity matrices for this coupling map (used by finesse configs)
        from .ablation import make_synthetic_fidelity, make_uniform_fidelity
        F_synth   = make_synthetic_fidelity(cm, seed=42)
        F_uniform = make_uniform_fidelity(cm)

        for mult in gate_multipliers:
            n_gates = n_qubits * mult
            label_base = f"clifford_{n_qubits}q_{n_gates}g"
            print(f"\n--- {label_base}  (phys={n_phys}, grid topology) ---")
            for cfg_label, cfg_kwargs in cfg_table.items():
                # Resolve fidelity_matrix sentinels
                kwargs = dict(cfg_kwargs)
                fm = kwargs.get('fidelity_matrix')
                if fm is True:
                    kwargs['fidelity_matrix'] = F_synth
                elif fm is False or fm is None:
                    kwargs.pop('fidelity_matrix', None)
                # (array values are used as-is)

                cfg_ok = True
                for seed in range(n_seeds):
                    qc = random_clifford_circuit(n_qubits, n_gates, seed=seed)
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        dag_phys = apply_trivial_layout(qc.copy(), cm)
                    rd, _, fc = route(
                        copy.deepcopy(dag_phys), cm,
                        seed=seed, **kwargs,
                    )
                    rqc = dag_to_circuit(rd)
                    ok = check_routing(
                        qc, rqc, fc, cm_edges,
                        dag_phys=dag_phys,
                        label=f'{label_base}/{cfg_label}/s{seed}',
                        verify='statevector',
                        sv_max_qubits=sv_max_qubits,
                    )
                    cfg_ok = cfg_ok and ok
                    all_ok = all_ok and ok
                status = 'OK' if cfg_ok else 'FAIL'
                print(f"  {cfg_label:<18} {status}  ({n_seeds} seeds)")

    print()
    print('All checks passed \u2713' if all_ok else 'SOME CHECKS FAILED \u2717')
    return all_ok
