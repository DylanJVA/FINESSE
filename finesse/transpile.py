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
from qiskit.transpiler import CouplingMap, PassManager
from qiskit.transpiler.passes import VF2Layout

from qiskit.transpiler import PassManager as _PassManager

from .benchmarks import apply_trivial_layout, make_unroll_consolidate
from .mirror import circuit_lf_cost
from .routing import route, _layout_pass

# ── Deployed budget levers ────────────────────────────────────────────────────
# Trial budget ≈ n_seeds × (n_traversals + len(betas)+1) passes; SABRE ≈ 60.
# Current split (8 seeds, 4 finite β + ∞): 8 × (2 + 5) = 56 passes.
# Tune these two to trade layout diversity against β resolution.
DEFAULT_N_SEEDS = 8
_DEFAULT_BETAS  = [0, 10, 100, 1000]   # β=∞ appended in finesse_transpile → 5-pt grid
# (the dense β grid for the sensitivity figure is passed explicitly by
#  finesse_beta_sweep.py; it does not use this default.)


def _vf2_initial_cur(qc, coupling_map, n_phys, seed):
    """Try Qiskit VF2Layout for a perfect embedding of qc into the device.

    Returns an initial_cur (cur[orig] = physical qubit of logical qubit orig,
    length n_phys, ancillas filling leftover physicals), or None if VF2 finds
    no perfect subgraph isomorphism (routing-heavy circuits).
    """
    try:
        pm = PassManager([VF2Layout(coupling_map, seed=seed,
                                    call_limit=10**6, max_trials=100)])
        pm.run(qc)
        layout = pm.property_set.get('layout')
    except Exception:
        return None
    if layout is None:
        return None
    vb = layout.get_virtual_bits()          # {qubit: physical}
    n_orig = qc.num_qubits
    cur = [-1] * n_phys
    used = set()
    for q, phys in vb.items():
        i = qc.find_bit(q).index
        if i < n_orig:
            cur[i] = phys
            used.add(phys)
    leftover = [p for p in range(n_phys) if p not in used]
    j = 0
    for i in range(n_phys):
        if cur[i] == -1:
            cur[i] = leftover[j]
            j += 1
    return cur


def finesse_transpile(
    qc: QuantumCircuit,
    coupling_map: CouplingMap,
    fidelity_matrix: np.ndarray,
    basis_gate: str = 'sqrt_iswap',
    betas: list[float] | None = None,
    n_seeds: int = DEFAULT_N_SEEDS,
    n_traversals: int = 2,
    seed: int = 0,
    parallel: bool = True,
    use_vf2: bool = True,
    consolidate: bool = True,
    return_curve: bool = False,
    aggression: int = 2,
    use_fidelity: bool = True,
):
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
        n_seeds:        Number of distinct warmed layouts to try.
        n_traversals:   Number of SABRE layout warmup passes per seed, alternating
                        forward/backward (0 = no warmup, route from random layout;
                        2 = forward-backward; 3 = forward-backward-forward). The
                        final β routing pass is separate and always runs.
        seed:           Master RNG seed.
        parallel:       Run the β grid with threads (False = serial; use under an
                        outer process pool to avoid oversubscription).
        use_vf2:        Also try a Qiskit VF2Layout perfect-embedding as an extra
                        candidate layout (free win on embeddable circuits; VF2
                        finds nothing on routing-heavy ones).
        consolidate:    Unroll high-level gates and consolidate consecutive 2Q
                        gates into 2Q unitary blocks before routing (what mirror
                        absorption expects, and what makes lf_cost comparable to a
                        consolidated Qiskit baseline). Leave True unless qc is
                        already in 2Q-block form.

    Returns:
        (circuit, best_beta): best routed QuantumCircuit and the β that achieved it.
    """
    if consolidate:
        qc = _PassManager(make_unroll_consolidate()).run(qc)
    dag_phys = apply_trivial_layout(qc, coupling_map)
    n_phys   = coupling_map.size()
    n_virtual = dag_phys.num_qubits()
    rng = np.random.default_rng(seed)

    # β grid only when fidelity routing is active (FASST/FINESSE). Without it
    # (MIRAGE), there is no β to search: one hop-count route per layout.
    if use_fidelity:
        if betas is None:
            betas = _DEFAULT_BETAS
        grid_betas = list(betas) + [float('inf')]
    else:
        grid_betas = [None]

    # ── Generate n_seeds warmed layouts ───────────────────────────────────────
    # Each seed: n_traversals layout passes, alternating forward/backward
    # (t=0 forward, t=1 backward, t=2 forward, ...). emit_ops=False, free.
    layouts = []
    for _ in range(n_seeds):
        warmup_seed = int(rng.integers(2**31))
        initial = list(rng.permutation(n_phys)[:n_virtual])
        for t in range(n_traversals):
            initial = _layout_pass(dag_phys, coupling_map, initial,
                                   reverse=bool(t % 2), seed=warmup_seed)
        layouts.append((warmup_seed, initial))

    # ── VF2 perfect-embedding layout (extra candidate; None on routing-heavy) ──
    if use_vf2:
        vf2_cur = _vf2_initial_cur(qc, coupling_map, n_phys, seed)
        if vf2_cur is not None:
            layouts.append((int(rng.integers(2**31)), vf2_cur))

    # ── (layout, β) grid — all independent, run in parallel ──────────────────
    trials = [(ws, cur, beta) for ws, cur in layouts for beta in grid_betas]

    mirror_on = aggression > 0
    F_score = fidelity_matrix  # always score the result by log-infidelity

    def _trial(args):
        warmup_seed, initial_cur, beta = args
        if beta is None:
            # MIRAGE = FINESSE minus fidelity: hop-count routing with the same
            # routing-aware mirror (accept when it doesn't worsen hop-count),
            # not MIRAGE's gate-cost-only pure_mirror. aggression gates it.
            routed, _, _ = route(
                copy.deepcopy(dag_phys), coupling_map,
                seed=warmup_seed, initial_cur=list(initial_cur),
                mode='lightsabre', aggression=aggression,
                pure_mirror=False,
                basis_gate=basis_gate,
            )
        else:
            # FASST/FINESSE: fidelity-weighted routing; mirror (if on) is
            # fidelity-aware. β=∞ → α=0 (pure fidelity).
            alpha_r = 0.0 if math.isinf(beta) else 1.0
            beta_r  = 1.0 if math.isinf(beta) else beta
            F = fidelity_matrix.copy()
            routed, _, _ = route(
                copy.deepcopy(dag_phys), coupling_map,
                seed=warmup_seed, initial_cur=list(initial_cur),
                mode='lightsabre', aggression=aggression,
                fidelity_matrix=F,
                fidelity_mirror=mirror_on,
                basis_gate=basis_gate,
                alpha=alpha_r, beta=beta_r,
            )
        cost = circuit_lf_cost(routed, F_score, basis_gate=basis_gate)
        return routed, cost, beta

    # parallel=False: run trials serially (use when an outer process pool already
    # saturates the cores, to avoid thread oversubscription).
    if parallel:
        with ThreadPoolExecutor(max_workers=len(trials)) as pool:
            results = list(pool.map(_trial, trials))
    else:
        results = [_trial(t) for t in trials]

    best_dag, best_cost, best_beta = None, float('inf'), 0.0
    per_beta: dict[float, list[float]] = {}   # β → list of lf, one per layout
    for routed, cost, beta in results:
        if cost < best_cost:
            best_dag, best_cost, best_beta = routed, cost, beta
        per_beta.setdefault(beta, []).append(cost)

    if return_curve:
        # (β, [lf per layout]) sorted by β, with β=inf (pure fidelity) last.
        # For β sensitivity, aggregate these across layouts (mean); the search
        # itself uses the min, which is best_beta above.
        def _bkey(kv):
            b = kv[0]
            return math.inf if (b is None or math.isinf(b)) else b
        curve = sorted(per_beta.items(), key=_bkey)
        return dag_to_circuit(best_dag), best_beta, curve

    return dag_to_circuit(best_dag), best_beta
