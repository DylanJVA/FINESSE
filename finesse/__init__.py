# finesse/__init__.py
#
# Public API surface. Import specialized benchmark/ablation utilities directly
# from their submodules:
#   from finesse.benchmarks import check_routing, run_correctness_suite, ...
#   from finesse.ablation   import ABLATION_CONFIGS, make_synthetic_fidelity, ...
#   from finesse.mirror     import decomp_cost, weyl_coords, SWAP_MATRIX, ...

# ── Routing ──────────────────────────────────────────────────────────────────
from .routing    import route
from .inline_pass import InlineMirageSwap

# ── Qiskit pipeline passes ────────────────────────────────────────────────────
from .layout    import FidelityLayout, FinesseLayout
try:
    from .decompose import MirageDecompose
except ModuleNotFoundError:
    MirageDecompose = None

# ── Pipeline setup ────────────────────────────────────────────────────────────
from .benchmarks import make_unroll_consolidate, apply_trivial_layout

# ── Fidelity ──────────────────────────────────────────────────────────────────
from .fidelity import fidelity_matrix_from_backend, build_target_from_fidelities
from .mirror import circuit_lf_cost

# ── Benchmark essentials (commonly used in notebooks) ─────────────────────────
from .benchmarks import (
    make_tokyo, fetch_qasm, check_routing, evaluate_routing_checks,
    print_routing_check_report, run_correctness_suite, run_config_correctness_suite,
    run_clifford_correctness_suite, random_clifford_circuit,
    line_cm, grid_cm, prepare_dag, benchmark_mode, swap_count,
    strip_regs, permutation_correction_qc,
    fetch_qasmbench, available_redqueen_circuits, available_qasmbench_circuits,
    load_redqueen_circuits, load_qasmbench_circuits,
)
from .ablation import (
    make_synthetic_fidelity, make_uniform_fidelity,
    ABLATION_CONFIGS, ABLATION_LABELS, FIDELITY_CONFIGS,
    QISKIT_CONFIGS, NOTEBOOK_CONFIGS, QUICK_CIRCUITS, BENCH_CIRCUITS,
    run_ablation_correctness,
)
from .benchmarks import BenchmarkConfig, run_benchmark

__all__ = [
    # Routing
    "route",
    "InlineMirageSwap",
    # Passes
    "FidelityLayout",
    "FinesseLayout",
    "MirageDecompose",
    # Pipeline setup
    "make_unroll_consolidate",
    "apply_trivial_layout",
    # Fidelity
    "fidelity_matrix_from_backend",
    "build_target_from_fidelities",
    # Benchmark essentials
    "make_tokyo",
    "fetch_qasm",
    "fetch_qasmbench",
    "available_redqueen_circuits",
    "available_qasmbench_circuits",
    "load_redqueen_circuits",
    "load_qasmbench_circuits",
    "check_routing",
    "evaluate_routing_checks",
    "print_routing_check_report",
    "run_correctness_suite",
    "run_config_correctness_suite",
    "run_clifford_correctness_suite",
    "random_clifford_circuit",
    "line_cm",
    "grid_cm",
    "prepare_dag",
    "benchmark_mode",
    "swap_count",
    "strip_regs",
    "permutation_correction_qc",
    "make_synthetic_fidelity",
    "make_uniform_fidelity",
    "circuit_lf_cost",
    "BenchmarkConfig",
    "run_benchmark",
    "run_ablation_correctness",
    "QUICK_CIRCUITS",
    "BENCH_CIRCUITS",
    "ABLATION_CONFIGS",
    "ABLATION_LABELS",
    "FIDELITY_CONFIGS",
    "QISKIT_CONFIGS",
    "NOTEBOOK_CONFIGS",
]
