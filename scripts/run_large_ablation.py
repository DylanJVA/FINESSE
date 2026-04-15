"""
Large-scale ablation benchmark across all downloadable red-queen circuits.

Runs every circuit in circuits/redqueen/ that fits on Q20 Tokyo and has
between MIN_2Q and MAX_2Q two-qubit gates.  For each (circuit, config, seed)
the routing is attempted n_trials times and the best result is kept.

Results are written incrementally after each circuit so the run can be
interrupted and resumed (existing entries in the output file are skipped).

Usage:
    python scripts/run_large_ablation.py [--seeds N] [--trials N] [--out FILE]

Defaults: 5 seeds, 20 trials, ablation_large.json
"""
import argparse
import copy
import json
import os
import sys
import time
import warnings
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from qiskit.converters import dag_to_circuit
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import (
    ApplyLayout, EnlargeWithAncilla, FullAncillaAllocation,
    SabreSwap, TrivialLayout,
)

from finesse.ablation import (
    ABLATION_CONFIGS,
    ABLATION_LABELS,
    FIDELITY_CONFIGS,
    UNIFORM_FIDELITY_CONFIGS,
    QISKIT_CONFIGS,
    circuit_lf_cost,
    make_synthetic_fidelity,
    make_uniform_fidelity,
)
from finesse.benchmarks import apply_trivial_layout, check_routing, fetch_qasm, make_tokyo
from finesse.routing import route


def _run_qiskit_sabre(qc, cm, seed, n_trials):
    """Route qc with Qiskit's native Rust SabreSwap under trivial layout.

    Uses trivial layout (same starting point as our configs) so the comparison
    isolates routing quality.  n_trials is passed directly to SabreSwap so
    Qiskit's internal Rust loop selects the best of n_trials attempts.
    """
    pm = PassManager([
        TrivialLayout(cm),
        FullAncillaAllocation(cm),
        EnlargeWithAncilla(),
        ApplyLayout(),
        SabreSwap(cm, heuristic='decay', seed=seed, trials=n_trials),
    ])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return pm.run(qc.copy())

# ── Circuit filter ────────────────────────────────────────────────────────────
MIN_2Q = 30     # skip trivially small circuits
MAX_2Q = 3000   # skip circuits that would take too long per routing call

CIRCUITS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), 'circuits', 'redqueen'
)


def discover_circuits(cm):
    """Return list of (label, qasm_path) for circuits that pass the filter."""
    candidates = []
    for fname in sorted(os.listdir(CIRCUITS_DIR)):
        if not fname.endswith('.qasm'):
            continue
        fpath = os.path.join(CIRCUITS_DIR, fname)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                qc = fetch_qasm(fname)
            if qc.num_qubits > cm.size():
                continue
            two_q = sum(1 for inst in qc.data if inst.operation.num_qubits == 2)
            if two_q < MIN_2Q or two_q > MAX_2Q:
                continue
            label = fname.replace('.qasm', '')
            candidates.append((label, fname, qc.num_qubits, two_q))
        except Exception as e:
            print(f"  [skip] {fname}: {e}")
    return candidates


# ── Per-circuit benchmark ─────────────────────────────────────────────────────

def _best_routing(dag_phys, cm, kwargs, cfg_key, F, n_trials, base_seed):
    best_rd, best_swap, best_lf = None, float('inf'), float('inf')
    for t in range(n_trials):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            rd, _, _ = route(
                copy.deepcopy(dag_phys), cm,
                seed=base_seed * n_trials + t, **kwargs,
            )
        rqc = dag_to_circuit(rd)
        swaps = rqc.count_ops().get('swap', 0)
        lf    = circuit_lf_cost(rd, F) if cfg_key in FIDELITY_CONFIGS else None

        if cfg_key in FIDELITY_CONFIGS:
            if lf < best_lf:
                best_rd, best_swap, best_lf = rd, swaps, lf
        else:
            if swaps < best_swap:
                best_rd, best_swap, best_lf = rd, swaps, lf
    return best_rd, best_swap, best_lf


def run_circuit(label, fname, cm, F, F_uniform, n_seeds, n_trials, check_correctness=False):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        qc = fetch_qasm(fname)
        dag_phys = apply_trivial_layout(qc.copy(), cm)

    cm_edges = list(cm.get_edges())
    cfg_results = {}
    correctness: dict[str, bool] = {}
    all_correct = True

    for cfg_key, cfg_kwargs in ABLATION_CONFIGS.items():
        swap_counts, gate_depths, lf_costs = [], [], []

        if cfg_key in QISKIT_CONFIGS:
            # Qiskit native SabreSwap — trials handled internally by Rust
            for seed in range(n_seeds):
                rqc = _run_qiskit_sabre(qc, cm, seed, n_trials)
                swap_counts.append(rqc.count_ops().get('swap', 0))
                gate_depths.append(rqc.depth())
                lf_costs.append(None)
        else:
            kwargs = dict(cfg_kwargs)
            if cfg_key in FIDELITY_CONFIGS:
                kwargs['fidelity_matrix'] = F
            elif cfg_key in UNIFORM_FIDELITY_CONFIGS:
                kwargs['fidelity_matrix'] = F_uniform
            for seed in range(n_seeds):
                rd, swaps, lf = _best_routing(
                    dag_phys, cm, kwargs, cfg_key, F, n_trials, seed,
                )
                rqc = dag_to_circuit(rd)
                swap_counts.append(swaps)
                gate_depths.append(rqc.depth())
                lf_costs.append(lf)

                # Correctness: statevector check on first seed only
                if check_correctness and seed == 0:
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        rd_check, _, fc_check = route(
                            copy.deepcopy(dag_phys), cm, seed=0, **kwargs,
                        )
                    rqc_check = dag_to_circuit(rd_check)
                    ok = check_routing(
                        qc, rqc_check, fc_check, cm_edges,
                        dag_phys=dag_phys,
                        label=f'{label}/{cfg_key}',
                        verify='statevector',
                        sv_max_qubits=cm.size(),
                    )
                    correctness[cfg_key] = ok
                    if not ok:
                        all_correct = False

        cfg_results[cfg_key] = {
            'swap_counts': swap_counts,
            'gate_depths': gate_depths,
            'lf_costs':    lf_costs,
        }
        avg_swap = float(np.mean(swap_counts))
        lf_vals  = [c for c in lf_costs if c is not None]
        lf_str   = f"lf={np.mean(lf_vals):.2f}" if lf_vals else ""
        print(f"    {ABLATION_LABELS[cfg_key]:<22} swaps={avg_swap:.1f}  {lf_str}")

    if check_correctness and not all_correct:
        print(f"  *** CORRECTNESS FAILURES: {[k for k,v in correctness.items() if not v]}")

    result = {
        'label':      label,
        'n_logical':  qc.num_qubits,
        'n_physical': cm.size(),
        'results':    cfg_results,
    }
    if check_correctness:
        result['correctness'] = correctness
    return result


# ── Averages ──────────────────────────────────────────────────────────────────

def compute_averages(circuit_results):
    averages = {}
    for cfg_key in ABLATION_CONFIGS:
        avg_swap = float(np.mean([
            np.mean(c['results'][cfg_key]['swap_counts'])
            for c in circuit_results
        ]))
        avg_depth = float(np.mean([
            np.mean(c['results'][cfg_key]['gate_depths'])
            for c in circuit_results
        ]))
        lf_vals = [
            v for c in circuit_results
            for v in c['results'][cfg_key]['lf_costs']
            if v is not None
        ]
        # Store mean -log-fidelity cost directly — exp(-lf) underflows to zero
        # for circuits with many gates, so the raw lf_cost is the usable metric.
        # Lower is better (less total gate error).
        avg_lf_cost = float(np.mean(lf_vals)) if lf_vals else None
        averages[cfg_key] = {
            'label':        ABLATION_LABELS[cfg_key],
            'avg_swap':     avg_swap,
            'avg_depth':    avg_depth,
            'avg_lf_cost':  avg_lf_cost,
        }
    return averages


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Large-scale FINESSE ablation")
    parser.add_argument('--seeds',  type=int, default=5)
    parser.add_argument('--trials', type=int, default=20)
    parser.add_argument('--out',    default='ablation_large.json')
    parser.add_argument('--check-correctness', action='store_true',
                        help='Run statevector check on one seed per (circuit, config). '
                             'Slow (~5-30s per check) but confirms routing correctness.')
    args = parser.parse_args()

    cm = make_tokyo()
    F         = make_synthetic_fidelity(cm, seed=42)
    F_uniform = make_uniform_fidelity(cm)

    circuits = discover_circuits(cm)
    print(f"Found {len(circuits)} circuits passing filter "
          f"({MIN_2Q}–{MAX_2Q} 2Q gates, ≤{cm.size()} qubits)")
    for label, fname, nq, n2q in circuits:
        print(f"  {label:<30} {nq:>3}q  {n2q:>5} 2Q gates")

    # Load existing results for resumption
    existing: dict[str, dict] = {}
    if os.path.exists(args.out):
        with open(args.out) as f:
            prev = json.load(f)
        for c in prev.get('circuits', []):
            existing[c['label']] = c
        print(f"\nResuming: {len(existing)} circuits already done.")

    circuit_results = list(existing.values())
    done_labels = set(existing)

    total = len(circuits)
    t0 = time.time()

    for idx, (label, fname, nq, n2q) in enumerate(circuits, 1):
        if label in done_labels:
            print(f"\n[{idx}/{total}] {label} — already done, skipping.")
            continue

        print(f"\n[{idx}/{total}] {label}  ({nq}q, {n2q} 2Q gates, "
              f"{args.seeds} seeds × {args.trials} trials)")
        t_circ = time.time()

        result = run_circuit(label, fname, cm, F, F_uniform, args.seeds, args.trials,
                             check_correctness=args.check_correctness)
        circuit_results.append(result)

        elapsed = time.time() - t_circ
        total_elapsed = time.time() - t0
        done_count = idx - len([l for l in [label] if l in done_labels])
        print(f"  done in {elapsed:.1f}s  (total {total_elapsed/60:.1f} min)")

        # Write incrementally after each circuit
        averages = compute_averages(circuit_results)
        output = {
            'generated':     datetime.now().isoformat(timespec='seconds'),
            'n_seeds':       args.seeds,
            'n_trials':      args.trials,
            'n_circuits':    len(circuit_results),
            'circuit_filter': {'min_2q': MIN_2Q, 'max_2q': MAX_2Q},
            'config_order':  list(ABLATION_CONFIGS.keys()),
            'config_labels': ABLATION_LABELS,
            'averages':      averages,
            'circuits':      circuit_results,
            'fidelity_matrix': F.tolist(),
        }
        with open(args.out, 'w') as f:
            json.dump(output, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Done — {len(circuit_results)} circuits in "
          f"{(time.time()-t0)/60:.1f} min")
    print(f"Saved to {args.out}")
    print()
    print(f"{'Config':<22} {'Avg SWAPs':>10} {'vs SABRE':>10} {'Avg LF cost':>14}")
    print("-" * 60)
    averages = compute_averages(circuit_results)
    sabre_swap = averages['sabre']['avg_swap']
    for cfg_key, avg in averages.items():
        pct  = (avg['avg_swap'] - sabre_swap) / sabre_swap * 100
        lf_s = f"{avg['avg_lf_cost']:.3f}" if avg['avg_lf_cost'] is not None else '  n/a'
        pct_s = '    —' if cfg_key == 'sabre' else f'{pct:+.1f}%'
        print(f"{avg['label']:<22} {avg['avg_swap']:>10.1f} {pct_s:>10} {lf_s:>14}")


if __name__ == '__main__':
    main()
