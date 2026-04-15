"""
Compute circuit_lf_cost(F_uniform) for every routing config.

Routes each benchmark circuit with all 8 configs and measures lf_cost under
a uniform fidelity matrix so all configs can be compared on the same axis.
With uniform F, lf_cost is proportional to total native 2Q gate count
(sum of decomp_cost k over all 2Q gates), which is the honest single metric
that accounts for both explicit SWAPs (k=3) and mirror gate overhead.

Usage:
    python scripts/run_lf_uniform_ablation.py [--seeds N] [--trials N] [--out lf_uniform.json]
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

from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import (
    ApplyLayout, EnlargeWithAncilla, FullAncillaAllocation,
    SabreSwap, TrivialLayout,
)

from finesse.ablation import (
    ABLATION_CONFIGS, ABLATION_LABELS,
    FIDELITY_CONFIGS, UNIFORM_FIDELITY_CONFIGS, QISKIT_CONFIGS,
    circuit_lf_cost, make_synthetic_fidelity, make_uniform_fidelity,
)
from finesse.benchmarks import apply_trivial_layout, fetch_qasm, make_tokyo
from finesse.routing import route


CIRCUITS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), 'circuits', 'redqueen'
)
MIN_2Q, MAX_2Q = 30, 3000


def discover_circuits(cm):
    candidates = []
    for fname in sorted(os.listdir(CIRCUITS_DIR)):
        if not fname.endswith('.qasm'):
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                qc = fetch_qasm(fname)
            if qc.num_qubits > cm.size():
                continue
            two_q = sum(1 for inst in qc.data if inst.operation.num_qubits == 2)
            if two_q < MIN_2Q or two_q > MAX_2Q:
                continue
            candidates.append((fname.replace('.qasm', ''), fname, qc.num_qubits, two_q))
        except Exception as e:
            print(f'  [skip] {fname}: {e}')
    return candidates


def _run_qiskit_sabre(qc, cm, seed, n_trials):
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


def _best_routing(dag_phys, cm, kwargs, cfg_key, F_uniform, n_trials, base_seed):
    """Run n_trials, return best by swap count (all configs use same criterion here)."""
    best_rd, best_swap = None, float('inf')
    for t in range(n_trials):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            rd, _, _ = route(
                copy.deepcopy(dag_phys), cm,
                seed=base_seed * n_trials + t, **kwargs,
            )
        swaps = dag_to_circuit(rd).count_ops().get('swap', 0)
        if swaps < best_swap:
            best_rd, best_swap = rd, swaps
    return best_rd, best_swap


def run_circuit(label, fname, cm, F_uniform, F_synth, n_seeds, n_trials):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        qc = fetch_qasm(fname)
        dag_phys = apply_trivial_layout(qc.copy(), cm)

    cfg_results = {}
    for cfg_key, cfg_kwargs in ABLATION_CONFIGS.items():
        swap_counts, lf_uniform_costs, lf_synth_costs = [], [], []

        if cfg_key in QISKIT_CONFIGS:
            for seed in range(n_seeds):
                rqc = _run_qiskit_sabre(qc, cm, seed, n_trials)
                rd = circuit_to_dag(rqc)
                swap_counts.append(rqc.count_ops().get('swap', 0))
                lf_uniform_costs.append(float(circuit_lf_cost(rd, F_uniform)))
                lf_synth_costs.append(None)
        else:
            kwargs = dict(cfg_kwargs)
            # All configs get F_uniform for routing (this levels the playing field)
            if cfg_key in FIDELITY_CONFIGS:
                kwargs['fidelity_matrix'] = F_synth   # real fidelity for routing
            elif cfg_key in UNIFORM_FIDELITY_CONFIGS:
                kwargs['fidelity_matrix'] = F_uniform

            for seed in range(n_seeds):
                rd, swaps = _best_routing(
                    dag_phys, cm, kwargs, cfg_key, F_uniform, n_trials, seed,
                )
                swap_counts.append(swaps)
                # lf_cost under uniform F for all configs — the honest single metric
                lf_uniform_costs.append(float(circuit_lf_cost(rd, F_uniform)))
                # lf_cost under real F only for fidelity-aware configs
                if cfg_key in FIDELITY_CONFIGS:
                    lf_synth_costs.append(float(circuit_lf_cost(rd, F_synth)))
                else:
                    lf_synth_costs.append(None)

        cfg_results[cfg_key] = {
            'swap_counts':      swap_counts,
            'lf_uniform_costs': lf_uniform_costs,
            'lf_synth_costs':   lf_synth_costs,
        }
        avg_sw = float(np.mean(swap_counts))
        avg_lf = float(np.mean(lf_uniform_costs))
        print(f'    {ABLATION_LABELS[cfg_key]:<22} swaps={avg_sw:.1f}  lf(unif)={avg_lf:.2f}')

    return {
        'label':      label,
        'n_logical':  qc.num_qubits,
        'n_physical': cm.size(),
        'results':    cfg_results,
    }


def compute_averages(circuit_results):
    averages = {}
    for cfg_key in ABLATION_CONFIGS:
        avg_swap = float(np.mean([
            np.mean(c['results'][cfg_key]['swap_counts'])
            for c in circuit_results
        ]))
        lf_u_vals = [
            v for c in circuit_results
            for v in c['results'][cfg_key]['lf_uniform_costs']
            if v is not None
        ]
        lf_s_vals = [
            v for c in circuit_results
            for v in c['results'][cfg_key]['lf_synth_costs']
            if v is not None
        ]
        averages[cfg_key] = {
            'label':           ABLATION_LABELS[cfg_key],
            'avg_swap':        avg_swap,
            'avg_lf_uniform':  float(np.mean(lf_u_vals)) if lf_u_vals else None,
            'avg_lf_synth':    float(np.mean(lf_s_vals)) if lf_s_vals else None,
        }
    return averages


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds',  type=int, default=3)
    parser.add_argument('--trials', type=int, default=5)
    parser.add_argument('--out',    default='lf_uniform.json')
    args = parser.parse_args()

    cm = make_tokyo()
    F_uniform = make_uniform_fidelity(cm)
    F_synth   = make_synthetic_fidelity(cm, seed=42)

    circuits = discover_circuits(cm)
    print(f'Found {len(circuits)} circuits')

    existing = {}
    if os.path.exists(args.out):
        with open(args.out) as f:
            prev = json.load(f)
        for c in prev.get('circuits', []):
            existing[c['label']] = c
        print(f'Resuming: {len(existing)} already done.')

    circuit_results = list(existing.values())
    done_labels = set(existing)
    total = len(circuits)
    t0 = time.time()

    for idx, (label, fname, nq, n2q) in enumerate(circuits, 1):
        if label in done_labels:
            print(f'\n[{idx}/{total}] {label} — skip')
            continue
        print(f'\n[{idx}/{total}] {label}  ({nq}q, {n2q} 2Q, {args.seeds}s×{args.trials}t)')
        t1 = time.time()
        result = run_circuit(label, fname, cm, F_uniform, F_synth, args.seeds, args.trials)
        circuit_results.append(result)
        print(f'  done in {time.time()-t1:.1f}s')

        averages = compute_averages(circuit_results)
        with open(args.out, 'w') as f:
            json.dump({
                'generated':      datetime.now().isoformat(timespec='seconds'),
                'n_seeds':        args.seeds,
                'n_trials':       args.trials,
                'n_circuits':     len(circuit_results),
                'config_order':   list(ABLATION_CONFIGS.keys()),
                'config_labels':  ABLATION_LABELS,
                'averages':       averages,
                'circuits':       circuit_results,
            }, f, indent=2)

    print(f'\n{"="*60}')
    print(f'Done — {len(circuit_results)} circuits in {(time.time()-t0)/60:.1f} min\n')

    averages = compute_averages(circuit_results)
    sabre_sw = averages['sabre']['avg_swap']
    sabre_lf = averages['sabre']['avg_lf_uniform']
    print(f'{"Config":<22} {"Avg SWAPs":>10} {"vs SABRE":>9} {"lf(uniform F)":>14} {"vs SABRE":>9}')
    print('-' * 70)
    for k, v in averages.items():
        sw_pct = '    —' if k == 'sabre' else f'{(v["avg_swap"]-sabre_sw)/sabre_sw*100:+.1f}%'
        lf_pct = '    —' if k == 'sabre' else f'{(v["avg_lf_uniform"]-sabre_lf)/sabre_lf*100:+.1f}%'
        lf_str = f'{v["avg_lf_uniform"]:.2f}' if v['avg_lf_uniform'] else ' n/a'
        print(f'{v["label"]:<22} {v["avg_swap"]:>10.1f} {sw_pct:>9} {lf_str:>14} {lf_pct:>9}')


if __name__ == '__main__':
    main()
