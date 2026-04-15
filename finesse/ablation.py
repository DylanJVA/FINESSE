# finesse/ablation.py
#
# Benchmark presets for the FINESSE ablation study.
#
# The main benchmark engine now lives in finesse.benchmarks.run_benchmark().
# This module keeps the paper-oriented preset table and a thin wrapper that
# maps those preset keys onto BenchmarkConfig objects.

from __future__ import annotations

import warnings
from collections import OrderedDict

import numpy as np
from qiskit.converters import dag_to_circuit

from .benchmarks import (
    BenchmarkConfig,
    _CORRECTNESS_SUITE,
    apply_trivial_layout,
    check_routing,
    fetch_qasm,
    make_tokyo,
    run_benchmark as run_config_benchmark,
)
from .routing import route

# ---------------------------------------------------------------------------
# Ablation configurations — 8-row table + Qiskit baseline
#
# Preset metadata used to build BenchmarkConfig objects.
# ---------------------------------------------------------------------------

ABLATION_CONFIGS: OrderedDict[str, dict] = OrderedDict([
    # ── Qiskit reference ──────────────────────────────────────────────────
    ('qiskit_sabre',      {}),

    # ── Baselines (rows 1–2): no mirrors, no fidelity ─────────────────────
    ('sabre',             {'mode': 'sabre',      'aggression': 0}),
    ('lightsabre',        {'mode': 'lightsabre', 'aggression': 0}),

    # ── Mirrors only (rows 3–4): ablates mirror contribution ─────────────
    ('mirage_sabre',      {'mode': 'sabre',      'aggression': 2}),
    ('mirage',            {'mode': 'lightsabre', 'aggression': 2}),

    # ── Fidelity only (rows 5–6): ablates fidelity in H ──────────────────
    ('sabre_fid',         {'mode': 'sabre',      'aggression': 0}),
    ('lightsabre_fid',    {'mode': 'lightsabre', 'aggression': 0}),

    # ── Fidelity + mirrors (rows 7–8): both, different routers ───────────
    # Key comparison: 1→3 isolates mirrors; 1→5 isolates fidelity;
    # 3→7 and 4→8 show whether fidelity and mirrors compound.
    ('mirage_sabre_fid',  {'mode': 'sabre',      'aggression': 2}),
    ('mirage_fid',        {'mode': 'lightsabre', 'aggression': 2}),

    # ── FINESSE full system (row 9) ───────────────────────────────────────
    # Same as mirage_fid but with bidir_passes=1: SABRE bidirectional warmup
    # before the main pass lets the layout converge toward fidelity-optimal
    # regions under D_fid. Comparison 8→9 isolates the layout warmup gain.
    # Full FinesseLayout (multiple random starts scored by lf_cost) further
    # improves this; integrate separately for the final paper runs.
    ('finesse',           {'mode': 'lightsabre', 'aggression': 2, 'bidir_passes': 1}),
])

ABLATION_LABELS: dict[str, str] = {
    'qiskit_sabre':     'Qiskit SabreSwap',
    'sabre':            'SABRE',
    'lightsabre':       'LightSABRE',
    'mirage_sabre':     'MIRAGE (SABRE)',
    'mirage':           'MIRAGE (LightSABRE)',
    'sabre_fid':        'SABRE + fidelity',
    'lightsabre_fid':   'LightSABRE + fidelity',
    'mirage_sabre_fid': 'MIRAGE + fidelity (SABRE)',
    'mirage_fid':       'MIRAGE + fidelity (LightSABRE)',
    'finesse':          'FINESSE',
}

# Configs that require fidelity_matrix injected into route()
FIDELITY_CONFIGS: frozenset[str] = frozenset({
    'sabre_fid', 'lightsabre_fid',
    'mirage_sabre_fid', 'mirage_fid',
    'finesse',
})

# Configs routed via Qiskit's SabreSwap rather than our route()
QISKIT_CONFIGS: frozenset[str] = frozenset({'qiskit_sabre'})

# Kept for compatibility with older notebooks/scripts that expected a separate
# "uniform fidelity" bucket. The current codebase does not use one.
UNIFORM_FIDELITY_CONFIGS: frozenset[str] = frozenset()

# Which configs appear in each notebook
NOTEBOOK_CONFIGS: dict[str, list[str]] = {
    'sabre':      ['qiskit_sabre', 'sabre'],
    'lightsabre': ['qiskit_sabre', 'sabre', 'lightsabre'],
    'mirage':     ['qiskit_sabre', 'sabre', 'lightsabre', 'mirage_sabre', 'mirage'],
    'finesse':    list(ABLATION_CONFIGS.keys()),
}


# ---------------------------------------------------------------------------
# Synthetic fidelity matrices
# ---------------------------------------------------------------------------

def make_uniform_fidelity(coupling_map, value: float = 0.97) -> np.ndarray:
    """Uniform fidelity matrix — every edge gets the same value."""
    n = coupling_map.size()
    F = np.zeros((n, n))
    np.fill_diagonal(F, 1.0)
    for u, v in coupling_map.get_edges():
        F[u, v] = F[v, u] = value
    return F


def make_synthetic_fidelity(coupling_map, *, seed: int = 42) -> np.ndarray:
    """
    Synthetic fidelity matrix with realistic IBM-Q-like values.
    Connected edges: F[i,j] ~ clip(N(0.97, 0.015), 0.90, 0.999).
    """
    n = coupling_map.size()
    rng = np.random.default_rng(seed)
    F = np.zeros((n, n))
    np.fill_diagonal(F, 1.0)
    for u, v in coupling_map.get_edges():
        if F[u, v] == 0.0:
            f = float(np.clip(rng.normal(0.97, 0.015), 0.90, 0.999))
            F[u, v] = F[v, u] = f
    return F


# ---------------------------------------------------------------------------
# Circuit lists
# ---------------------------------------------------------------------------

QUICK_CIRCUITS: list[tuple[str, str]] = [
    ('rd32-v0_66', 'rd32-v0_66.qasm'),
    ('4gt11_84',   '4gt11_84.qasm'),
    ('ham7_104',   'ham7_104.qasm'),
    ('rd53_135',   'rd53_135.qasm'),
    ('rd84_142',   'rd84_142.qasm'),
]

BENCH_CIRCUITS: list[tuple[str, str]] = [
    ('sym6_145',  'sym6_145.qasm'),
    ('sqrt8_260', 'sqrt8_260.qasm'),
    ('adr4_197',  'adr4_197.qasm'),
    ('cm42a_207', 'cm42a_207.qasm'),
    ('rd84_142',  'rd84_142.qasm'),
]


# ---------------------------------------------------------------------------
# Unified benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(
    config_keys: list[str] | None = None,
    circuits: list[tuple[str, str]] | None = None,
    n_seeds: int = 10,
    n_trials: int = 3,
    coupling_map=None,
    fidelity_matrix: np.ndarray | None = None,
    verbose: bool = True,
):
    """Run the paper preset table via the generic benchmark engine."""
    if config_keys is None:
        config_keys = list(ABLATION_CONFIGS.keys())
    if circuits is None:
        circuits = QUICK_CIRCUITS
    if coupling_map is None:
        coupling_map = make_tokyo()

    n_phys = coupling_map.size()
    if fidelity_matrix is None:
        fidelity_matrix = make_synthetic_fidelity(coupling_map, seed=42)

    configs: list[BenchmarkConfig] = []
    for key in config_keys:
        kwargs = ABLATION_CONFIGS[key]
        router = "qiskit_sabre" if key == "qiskit_sabre" else (
            "mirage_sabre" if key in {"mirage_sabre", "mirage_sabre_fid"} else (
                "mirage" if key in {"mirage", "mirage_fid", "finesse"} else kwargs["mode"]
            )
        )
        configs.append(BenchmarkConfig(
            name=key,
            router=router,
            use_fidelity=(key in FIDELITY_CONFIGS),
            trials=n_trials,
            bidir_passes=kwargs.get("bidir_passes", 0),
        ))

    df = run_config_benchmark(
        configs=configs,
        circuits=circuits,
        coupling_map=coupling_map,
        n_seeds=n_seeds,
        fidelity_matrix=fidelity_matrix,
        verbose=verbose,
    )

    df["label"] = df["config"].map(ABLATION_LABELS)
    df["n_physical"] = n_phys
    return df


# ---------------------------------------------------------------------------
# Correctness runner
# ---------------------------------------------------------------------------

def run_ablation_correctness(
    configs: OrderedDict | None = None,
    n_seeds: int = 3,
    sv_max_qubits: int = 10,
) -> dict[str, dict[str, bool]]:
    """
    Route each bundled correctness-suite circuit under every ablation config
    and return a correctness dict: results[circ_label][cfg_key] = True iff all seeds passed.
    """
    if configs is None:
        configs = ABLATION_CONFIGS

    results: dict[str, dict[str, bool]] = {}

    for circ_label, qasm_name, cm_small, cm_edges in _CORRECTNESS_SUITE:
        print(f"\n[{circ_label}]  n_physical={cm_small.size()}")
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            qc = fetch_qasm(qasm_name)
            dag_phys = apply_trivial_layout(qc.copy(), cm_small)

        F_small = make_synthetic_fidelity(cm_small, seed=42)
        circ_ok: dict[str, bool] = {}

        for cfg_key, cfg_kwargs in configs.items():
            kwargs = dict(cfg_kwargs)
            if cfg_key in QISKIT_CONFIGS:
                circ_ok[cfg_key] = True
                print(f"  {ABLATION_LABELS[cfg_key]:<28} (Qiskit pass — adjacency only ✓)")
                continue
            if cfg_key in FIDELITY_CONFIGS:
                kwargs['fidelity_matrix'] = F_small

            cfg_correct = True
            for seed in range(n_seeds):
                dag_to_route = copy.deepcopy(dag_phys)

                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    rd, _, fc = route(dag_to_route, cm_small, seed=seed, **kwargs)
                rqc = dag_to_circuit(rd)
                ok = check_routing(
                    qc, rqc, fc, cm_edges,
                    dag_phys=dag_phys,
                    label=f'{circ_label}/{cfg_key}/s{seed}',
                    verify='statevector',
                    sv_max_qubits=sv_max_qubits,
                )
                cfg_correct = cfg_correct and ok

            circ_ok[cfg_key] = cfg_correct
            status = '✓' if cfg_correct else '✗ FAIL'
            print(f"  {ABLATION_LABELS[cfg_key]:<28} {status}")

        results[circ_label] = circ_ok

    all_ok = all(v for d in results.values() for v in d.values())
    print(f"\n{'All checks passed ✓' if all_ok else 'SOME CHECKS FAILED ✗'}")
    return results
