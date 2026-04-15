"""
Benchmark Our SABRE vs Qiskit on the full red-queen circuit suite.

Topology is chosen per-circuit: Q20 Tokyo for circuits that fit (≤ 20 qubits),
otherwise a square grid sized to the circuit. Results are saved after every
circuit so the run can be interrupted and resumed.

After `pip install -e .`:
    finesse-redqueen [--seeds N] [--out results/rq.json]
"""
import argparse
import copy
import json
import math
import os
import sys
import warnings
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from qiskit.converters import dag_to_circuit
from qiskit.transpiler import CouplingMap, PassManager
from qiskit.transpiler.passes import (
    ApplyLayout, EnlargeWithAncilla, FullAncillaAllocation,
    SabreSwap, TrivialLayout,
)

from finesse import apply_trivial_layout, fetch_qasm, make_tokyo
from finesse.benchmarks import _REDQUEEN_DIR
from finesse.routing import route

# ── Topology helpers ──────────────────────────────────────────────────────────

_TOKYO = make_tokyo()


def _grid_cm(rows: int, cols: int) -> CouplingMap:
    edges = []
    for i in range(rows):
        for j in range(cols):
            q = i * cols + j
            if j + 1 < cols:
                edges.append([q, q + 1])
            if i + 1 < rows:
                edges.append([q, q + cols])
    return CouplingMap(edges)


def topology_for(n_qubits: int) -> tuple[CouplingMap, str]:
    """Return the smallest standard topology that fits n_qubits."""
    if n_qubits <= _TOKYO.size():
        return _TOKYO, "Q20-Tokyo"
    side = math.ceil(math.sqrt(n_qubits))
    cm = _grid_cm(side, side)
    return cm, f"{side}x{side}-grid"


# ── Per-circuit benchmark ─────────────────────────────────────────────────────

def run_circuit(fname: str, seeds: int) -> dict:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            qc_raw = fetch_qasm(fname)
    except Exception as e:
        return {"error": f"load failed: {e}"}

    n_qubits = qc_raw.num_qubits
    n2q = sum(1 for inst in qc_raw.data if inst.operation.num_qubits == 2)
    cm, topo_label = topology_for(n_qubits)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dag_base = apply_trivial_layout(qc_raw.copy(), cm)

    our_swaps, qk_swaps = [], []
    for seed in range(seeds):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rd, _, _ = route(
                copy.deepcopy(dag_base), cm,
                aggression=0, seed=seed, mode="sabre",
            )
        our_swaps.append(dag_to_circuit(rd).count_ops().get("swap", 0))

        pm = PassManager([
            TrivialLayout(cm), FullAncillaAllocation(cm),
            EnlargeWithAncilla(), ApplyLayout(),
            SabreSwap(cm, heuristic="decay", seed=seed, trials=1),
        ])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            qk_swaps.append(pm.run(qc_raw.copy()).count_ops().get("swap", 0))

    return {
        "n_qubits": n_qubits,
        "n2q": n2q,
        "topology": topo_label,
        "our_swaps": our_swaps,
        "qk_swaps": qk_swaps,
        "our_mean": float(np.mean(our_swaps)),
        "qk_mean": float(np.mean(qk_swaps)),
        "our_std": float(np.std(our_swaps)),
        "qk_std": float(np.std(qk_swaps)),
    }


# ── Summary table ─────────────────────────────────────────────────────────────

def write_summary(results: dict, seeds: int, out_txt: str) -> None:
    rows   = [(f, r) for f, r in sorted(results.items()) if "our_mean" in r]
    errors = [(f, r) for f, r in sorted(results.items()) if "error" in r]

    lines = [
        "Red-Queen benchmark: Our SABRE vs Qiskit (LightSABRE)",
        f"Seeds: {seeds}  |  Generated: {datetime.now():%Y-%m-%d %H:%M}",
        "",
        f"{'Circuit':<30} {'Q':>3} {'2Q':>6} {'Topology':<14} {'Our SABRE':>12} {'Qiskit':>12} {'Δ':>7}",
        "-" * 88,
    ]
    our_all, qk_all = [], []
    for fname, r in rows:
        delta = r["our_mean"] - r["qk_mean"]
        lines.append(
            f"{fname:<30} {r['n_qubits']:>3} {r['n2q']:>6}"
            f" {r['topology']:<14}"
            f" {r['our_mean']:>9.1f}±{r['our_std']:.1f}"
            f" {r['qk_mean']:>9.1f}±{r['qk_std']:.1f}"
            f" {delta:>+7.1f}"
        )
        our_all.append(r["our_mean"])
        qk_all.append(r["qk_mean"])

    if our_all:
        lines += [
            "-" * 88,
            f"{'Mean over all circuits':<30} {'':>3} {'':>6} {'':>14}"
            f" {np.mean(our_all):>12.2f} {np.mean(qk_all):>12.2f}"
            f" {np.mean(our_all) - np.mean(qk_all):>+7.2f}",
            "",
            f"Circuits completed: {len(rows)}",
        ]
    if errors:
        lines.append(f"Errors ({len(errors)}): " + ", ".join(f for f, _ in errors))

    with open(out_txt, "w") as f:
        f.write("\n".join(lines) + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=20,
                        help="random seeds per circuit (default: 20)")
    parser.add_argument("--out", default="rq.json",
                        help="output JSON path (default: rq.json)")
    args = parser.parse_args()

    out_json = args.out
    out_txt  = out_json.replace(".json", ".txt")

    if not os.path.isdir(_REDQUEEN_DIR):
        sys.exit(f"circuits/ not found — run `finesse-download` first.")

    os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)

    results: dict = {}
    if os.path.exists(out_json):
        with open(out_json) as f:
            results = json.load(f)
        print(f"Resuming: {len(results)} circuits already done.", flush=True)

    circuits = sorted(f for f in os.listdir(_REDQUEEN_DIR) if f.endswith(".qasm"))
    print(f"{len(circuits)} circuits in suite.\n", flush=True)

    for i, fname in enumerate(circuits):
        if fname in results:
            print(f"[{i+1:3}/{len(circuits)}] {fname:<35} cached", flush=True)
            continue

        r = run_circuit(fname, args.seeds)
        results[fname] = r

        if "error" in r:
            print(f"[{i+1:3}/{len(circuits)}] {fname:<35} ERROR {r['error']}", flush=True)
        else:
            delta = r["our_mean"] - r["qk_mean"]
            print(
                f"[{i+1:3}/{len(circuits)}] {fname:<35}"
                f"  {r['topology']:<14}"
                f"  our={r['our_mean']:6.1f}  qk={r['qk_mean']:6.1f}  Δ={delta:+.1f}",
                flush=True,
            )

        with open(out_json, "w") as f:
            json.dump(results, f, indent=2)
        write_summary(results, args.seeds, out_txt)

    print(f"\nDone.  {out_json}  |  {out_txt}")


if __name__ == "__main__":
    main()
