# finesse/transpile.py
#
# Strategy: run a (seed × β) grid. Each seed gets its own bidir SABRE warmup
# to find a good initial layout, then is routed with every β value. For each
# (layout, β) the route is replayed swap_trials times with different random
# tie-breaks, keeping the best (this mirrors Qiskit SabreLayout's swap_trials).
# All trials are independent and run in parallel. Post-select by lf_cost.

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

from .benchmarks import apply_trivial_layout, make_unroll_consolidate, swap_count
from .mirror import circuit_lf_cost, circuit_2q_gate_count
from .routing import route, _layout_pass

# ── Deployed budget levers ────────────────────────────────────────────────────
# FINESSE exposes the same three search axes as Qiskit SabreLayout:
#   n_seeds      ~ layout_trials    distinct warmed initial layouts (keep best)
#   n_traversals ~ 2*max_iterations forward/backward warmup passes per layout
#   swap_trials  ~ swap_trials      re-route a fixed layout with random tie-breaks,
#                                   keep the best (SabreSwap breaks ties randomly)
# Routing-pass budget ≈ n_seeds × (n_traversals + len(grid_betas) × swap_trials).
#
# swap_trials only helps HOP-COUNT routers: integer hop scores tie constantly, so
# Qiskit routes 20× and keeps the lucky low-swap outcome. FINESSE's fidelity cost
# is continuous (a near-total order; ties < SCORE_EPSILON essentially never occur),
# so a single route already lands the best -- swap_trials>1 leaves the fidelity
# result unchanged (verified). We therefore deploy swap_trials=1 for fidelity
# routing and give the hop-count baseline (MIRAGE) swap_trials=20 to match SABRE.
# Deployed FINESSE (12 seeds, 2 warmup, 3-pt β grid, swap_trials=1): 12×(2+3) = 60.
DEFAULT_N_SEEDS     = 12
DEFAULT_SWAP_TRIALS = 1
_DEFAULT_BETAS      = [0, 100]         # β=∞ appended in finesse_transpile → 3-pt grid
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


# Initial-layout region size = REGION_FACTOR × circuit qubits (capped at device).
# A full permutation scatters a small circuit across a large device (25 qubits
# over 127), which the warm-up cannot undo; a *tight* region (1×) traps qubits in
# a low-fidelity corner. ~3× is the compromise: ≈ whole device when the circuit
# fills it (SNAIL), a connected neighbourhood with fidelity slack when it does
# not (IBM).
REGION_FACTOR = 3


def _compact_initial_cur(coupling_map, n_phys, n_active, rng):
    """Random initial layout whose active qubits occupy a connected region.

    Grow a connected BFS region of ~REGION_FACTOR×n_active physical qubits from a
    random start and place the active logical qubits within it; idle ancillas take
    whatever is left. The random start keeps layouts diverse across seeds.
    """
    region_size = min(n_phys, int(np.ceil(REGION_FACTOR * n_active)))
    adj = [[] for _ in range(n_phys)]
    for i, j in coupling_map.get_edges():
        adj[i].append(j); adj[j].append(i)

    start = int(rng.integers(n_phys))
    region, seen, frontier = [start], {start}, [start]
    while len(region) < region_size and frontier:
        nxt = []
        for u in frontier:
            for v in adj[u]:
                if v not in seen:
                    seen.add(v); region.append(v); nxt.append(v)
                    if len(region) >= region_size:
                        break
            if len(region) >= region_size:
                break
        frontier = nxt
    region = region[:region_size]

    rest = [p for p in range(n_phys) if p not in set(region)]
    rest = [int(x) for x in rng.permutation(rest)] if rest else []
    while len(region) < region_size and rest:       # disconnected map fallback
        region.append(rest.pop())
    region = [int(x) for x in rng.permutation(region)]
    return region + rest


def finesse_transpile(
    qc: QuantumCircuit,
    coupling_map: CouplingMap,
    fidelity_matrix: np.ndarray,
    basis_gate: str = 'sqrt_iswap',
    betas: list[float] | None = None,
    n_seeds: int = DEFAULT_N_SEEDS,
    n_traversals: int = 2,
    swap_trials: int = DEFAULT_SWAP_TRIALS,
    staged: bool = False,
    seed: int = 0,
    parallel: bool = True,
    use_vf2: bool = True,
    consolidate: bool = True,
    return_curve: bool = False,
    return_all: bool = False,
    aggression: int = 2,
    use_fidelity: bool = True,
    post_select: str = 'lf',
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
                        final β routing pass is separate and always runs. Analog of
                        Qiskit SabreLayout's 2*max_iterations layout passes.
        swap_trials:    Re-route each (layout, β) this many times with different
                        random SabreSwap tie-breaks and keep the best (by the
                        post_select metric). Analog of Qiskit SabreLayout's
                        swap_trials. 1 = single route (old behaviour, reproduced
                        exactly). Multiplies the routing budget by this factor.
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
    n_active  = min(qc.num_qubits, n_phys)   # real circuit qubits (rest are ancillas)
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
        initial = _compact_initial_cur(coupling_map, n_phys, n_active, rng)
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
    # seed index labels which warmed layout each trial came from.
    trials = [(i, ws, cur, beta)
              for i, (ws, cur) in enumerate(layouts) for beta in grid_betas]

    mirror_on = aggression > 0
    F_score = fidelity_matrix  # lf is always scored against the fidelity matrix
    # which recorded metric the post-selection minimizes
    _pskey = {'depth': 'depth', 'swaps': 'swaps', 'gates': 'gates'}.get(post_select, 'lf_cost')

    def _route_once(initial_cur, beta, route_seed):
        if beta is None:
            # MIRAGE = FINESSE minus fidelity: hop-count routing with the same
            # routing-aware mirror (accept when it doesn't worsen hop-count),
            # not MIRAGE's gate-cost-only pure_mirror. aggression gates it.
            routed, _, _ = route(
                copy.deepcopy(dag_phys), coupling_map,
                seed=route_seed, initial_cur=list(initial_cur),
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
                seed=route_seed, initial_cur=list(initial_cur),
                mode='lightsabre', aggression=aggression,
                fidelity_matrix=F,
                fidelity_mirror=mirror_on,
                basis_gate=basis_gate,
                alpha=alpha_r, beta=beta_r,
            )
        return routed

    def _make_rec(routed, seed_idx, beta):
        return {
            'seed': seed_idx, 'beta': beta,
            'swaps': swap_count(routed), 'depth': int(routed.depth()),
            'gates': circuit_2q_gate_count(routed, basis_gate=basis_gate),
            'lf_cost': circuit_lf_cost(routed, F_score, basis_gate=basis_gate),
        }

    def _trial(args):
        # GRID: for one (layout, β), replay SabreSwap swap_trials times with
        # different random tie-breaks and keep the best (by the post-select
        # metric). Every β pays the full swap search. k=0 reuses warmup_seed so
        # swap_trials=1 reproduces the old result exactly.
        seed_idx, warmup_seed, initial_cur, beta = args
        tie_rng = np.random.default_rng(warmup_seed)
        best_routed, best_rec = None, None
        for k in range(max(1, swap_trials)):
            route_seed = warmup_seed if k == 0 else int(tie_rng.integers(2**31))
            routed = _route_once(initial_cur, beta, route_seed)
            rec = _make_rec(routed, seed_idx, beta)
            if best_rec is None or rec[_pskey] < best_rec[_pskey]:
                best_routed, best_rec = routed, rec
        return best_routed, float(best_rec[_pskey]), beta, best_rec

    def _staged_trial(args):
        # STAGED: for one layout, scan every β once (swap_trials=1) to pick the
        # winning β, then spend the remaining swap_trials-1 replays only on that
        # β. Cost per layout = n_β + (swap_trials-1) instead of n_β × swap_trials.
        seed_idx, warmup_seed, initial_cur = args
        best_routed, best_rec = None, None
        for beta in grid_betas:                      # Stage 1: β scan, 1 route each
            routed = _route_once(initial_cur, beta, warmup_seed)
            rec = _make_rec(routed, seed_idx, beta)
            if best_rec is None or rec[_pskey] < best_rec[_pskey]:
                best_routed, best_rec = routed, rec
        win_beta = best_rec['beta']                  # Stage 2: refine the winner
        tie_rng = np.random.default_rng(warmup_seed + 1)
        for _k in range(max(1, swap_trials) - 1):
            routed = _route_once(initial_cur, win_beta, int(tie_rng.integers(2**31)))
            rec = _make_rec(routed, seed_idx, win_beta)
            if rec[_pskey] < best_rec[_pskey]:
                best_routed, best_rec = routed, rec
        return best_routed, float(best_rec[_pskey]), win_beta, best_rec

    # staged operates per-layout (one task each); grid per (layout, β).
    if staged:
        tasks = [(i, ws, cur) for i, (ws, cur) in enumerate(layouts)]
        run = _staged_trial
    else:
        tasks = trials
        run = _trial

    # parallel=False: run tasks serially (use when an outer process pool already
    # saturates the cores, to avoid thread oversubscription).
    if parallel:
        with ThreadPoolExecutor(max_workers=len(tasks)) as pool:
            results = list(pool.map(run, tasks))
    else:
        results = [run(t) for t in tasks]

    best_dag, best_cost, best_beta = None, float('inf'), 0.0
    per_beta: dict[float, list[float]] = {}   # β → list of lf, one per layout
    all_recs = []
    for routed, cost, beta, rec in results:
        if cost < best_cost:
            best_dag, best_cost, best_beta = routed, cost, beta
        per_beta.setdefault(beta, []).append(rec['lf_cost'])
        all_recs.append(rec)

    if return_all:
        # every trial's metrics; post-selection is done downstream (notebook)
        return dag_to_circuit(best_dag), best_beta, all_recs

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
